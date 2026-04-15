好，我给你一版可以直接交给 Codex 开做的 world model 设计。
我会基于你现在实际已有的数据来设计，不假设你已经有 root pose、contact、FK state 这些额外标注。

我先说结论：

在你当前只有
	•	38D robot action
	•	64D VAE latent
	•	text -> Qwen embedding

这三类数据时，最稳、最容易先跑通的一版是：
	•	C_t：从未来 latent 序列里提取的 coarse future plan tokens
	•	P_t：从未来 38D action 序列里提取的 future action dynamics summary tokens

也就是说，第一版里 P_t 我建议你把它定义为 “物理后果的代理量”，而不是强行定义成“真实世界状态”。这样更诚实，也更容易训稳。这个整体思路跟 MotionStreamer 的“文本 + 历史 latent → AR 条件 → diffusion 预测 next latent”、DIAL 的“latent intent bottleneck + warmup 再 joint train”、DreamVLA 的“预测紧凑且解耦的未来知识”、以及 WorldVLA 对 AR 误差传播问题的处理方向是对齐的。 ￼

⸻

0. 先约定数据形状

我这里默认你最后能整理成下面两种对齐后的序列：

A \in \mathbb{R}^{B \times T \times 38}
Z \in \mathbb{R}^{B \times T \times 64}

其中：
	•	A[:, t] 是第 t 帧的 38D 动作
	•	Z[:, t] 是第 t 帧对应的 64D latent

如果你现在磁盘里存的是 [B, T_chunk, 10, 64]，那就先 flatten 成 [B, T, 64]。
只要最终能做到“一帧一个 64D latent”，下面设计就可以直接用。

如果你的真实情况是“一个 latent 对应 10 帧”，也没关系，那就把下面所有的 t 理解成 block index，把 H 理解成未来几个 block，设计不变。

⸻

1. world model 的职责

你现有 actor 负责做：

z_t \sim \text{Actor}(text,\; z_{<t},\; \text{extra cond})

我建议 world model 改成负责先输出：

(C_t,\;P_t) = \text{WM}(text,\; z_{<t},\; a_{<t})

然后 actor 再做：

z_t \sim \text{Actor}(z_{<t},\; C_t,\; P_t,\; text)

这里：
	•	C_t：未来要做什么
	•	P_t：未来这段动作在 action space 里会怎样演化

所以 world model 不是直接输出下一帧 latent，而是先输出一组未来摘要 token。

⸻

2. 我建议的最终输出形式

2.1 C_t：4 个 plan tokens

定义：

C_t \in \mathbb{R}^{B \times M_c \times 64},\quad M_c=4

也就是 world model 预测 4 个 64 维 token。

这 4 个 token 的语义不是“4 个未来帧”，而是：
	•	C[0]：未来 latent chunk 的平均趋势
	•	C[1]：未来 latent chunk 的一阶低频变化
	•	C[2]：未来 latent chunk 的二阶低频变化
	•	C[3]：未来 latent chunk 的更细一点的低频模式

最关键的一点是：

C_t 的监督目标不要用可学习 teacher encoder，第一版直接用固定的 DCT/低频基来从 future latent chunk 构造。

这是我最推荐你现在就做的，因为：
	•	不会 target collapse
	•	不需要额外 teacher 网络
	•	语义清楚
	•	很适合 motion latent 这种时序信号

具体定义

取未来窗口：

Z^{fut}_t = [z_t, z_{t+1}, \dots, z_{t+H-1}] \in \mathbb{R}^{B \times H \times 64}

设 H=8。

构造一个固定的 DCT basis：

B_c \in \mathbb{R}^{4 \times 8}

取 DCT 的前 4 个低频基。然后：

C_t^\* = B_c \cdot Z^{fut}_t

实现上就是：

# z_fut: [B, H, 64]
# Bc:    [4, H]
c_star = torch.einsum('mh,bhd->bmd', Bc, z_fut)   # [B, 4, 64]

这就是你的 plan target。

⸻

2.2 P_t：4 个 dynamics tokens

定义：

P_t \in \mathbb{R}^{B \times M_p \times 38},\quad M_p=4

也就是预测 4 个 38 维 token。

这里我建议不要做成 DCT latent 的翻版，而是做成未来 action chunk 的动态统计摘要。
这样它和 C_t 真正分工不同。

具体定义

取未来动作窗口：

A^{fut}_t = [a_t, a_{t+1}, \dots, a_{t+H-1}] \in \mathbb{R}^{B \times H \times 38}

然后定义 4 个 token：

Token 1：未来动作均值

p^{(1)}_t = \frac{1}{H}\sum_{i=0}^{H-1} a_{t+i}

Token 2：未来动作净变化

p^{(2)}_t = a_{t+H-1} - a_t

Token 3：未来动作一阶差分均值

p^{(3)}_t = \frac{1}{H-1}\sum_{i=0}^{H-2}(a_{t+i+1}-a_{t+i})

Token 4：未来动作二阶差分均值

p^{(4)}_t = \frac{1}{H-2}\sum_{i=0}^{H-3}(a_{t+i+2}-2a_{t+i+1}+a_{t+i})

最后：

P_t^\* = [p^{(1)}_t,\; p^{(2)}_t,\; p^{(3)}_t,\; p^{(4)}_t]
\in \mathbb{R}^{B \times 4 \times 38}

代码就是：

def build_p_target(a_fut):   # [B, H, 38]
    p1 = a_fut.mean(dim=1)                     # [B, 38]
    p2 = a_fut[:, -1] - a_fut[:, 0]           # [B, 38]

    da = a_fut[:, 1:] - a_fut[:, :-1]         # [B, H-1, 38]
    p3 = da.mean(dim=1)                       # [B, 38]

    if a_fut.shape[1] >= 3:
        d2a = da[:, 1:] - da[:, :-1]          # [B, H-2, 38]
        p4 = d2a.mean(dim=1)                  # [B, 38]
    else:
        p4 = torch.zeros_like(p1)

    p_star = torch.stack([p1, p2, p3, p4], dim=1)  # [B, 4, 38]
    return p_star

这 4 个 token 对应的直觉分别是：
	•	P[0]：未来动作块大概会保持在哪个动作区域
	•	P[1]：这段未来动作的整体偏移方向
	•	P[2]：未来动作块平均“速度”
	•	P[3]：未来动作块平均“曲率/加速度”

所以 P_t 在第一版里不是“真实 physics”，而是future action dynamics summary。
这已经足够形成和 C_t 不同的结构角色：
	•	C_t 看的是 future latent plan
	•	P_t 看的是 future action dynamics

⸻

3. 为什么我强烈建议第一版这样做

因为你现在没有额外状态。
如果你硬定义 P_t 为 root displacement、contact、phase、COM，那就是在幻想自己有监督。

但现在这种定义完全建立在你手头真实有的数据上：
	•	C_t* 直接来自 future latent chunk
	•	P_t* 直接来自 future action chunk

不依赖额外标注，不依赖 simulator rollout，不依赖 FK。

而且这个结构是有理论直觉的：
	•	MotionStreamer 说明 continuous causal latent + AR diffusion 是成立的，并且历史 latent 对 next latent 很关键。 ￼
	•	DIAL 说明“先预测 latent future intent，再由执行头做 inverse dynamics”这种 bottleneck 设计是有效的。 ￼
	•	DreamVLA 说明 world knowledge 应该是紧凑且分解的，不是直接重建原始未来。 ￼
	•	WorldVLA 指出 AR action generation 会有误差传播，所以你的 world model 也要考虑训练时的鲁棒性。 ￼

⸻

4. world model 的输入

我建议 world model 单独作为一个模块，输入三项：

4.1 文本 token

来自 Qwen 的 hidden states：

E_x \in \mathbb{R}^{B \times L_x \times D_x}

文本压缩

不要直接全喂 world model。
先用 8 个 learnable text queries 压成 8 个文本条件 token：

T_x = \text{TextCompressor}(E_x) \in \mathbb{R}^{B \times 8 \times d}

实现就是 cross-attention pooling。

⸻

4.2 历史 latent

取过去 K 帧：

Z^{past}_t = [z_{t-K}, \dots, z_{t-1}] \in \mathbb{R}^{B \times K \times 64}

建议：

K = 32

⸻

4.3 历史 action

同样取过去 K 帧：

A^{past}_t = [a_{t-K}, \dots, a_{t-1}] \in \mathbb{R}^{B \times K \times 38}

⸻

5. world model 主体结构

我建议你直接做成下面这个。

5.1 per-step 融合 token

每个时间步，把过去 action 和 latent 融合成一个 motion token：

m_i = \text{MLP}_{in}([z_i; a_i]) \in \mathbb{R}^{d}

其中：

[z_i; a_i] \in \mathbb{R}^{102}

推荐：
	•	d_model = 512
	•	MLP_in: 102 -> 256 -> 512

于是得到：

M_t \in \mathbb{R}^{B \times K \times 512}

⸻

5.2 历史编码器

把 8 个文本 token + K 个 motion token 拼起来：

X_t = [T_x ; M_t] \in \mathbb{R}^{B \times (8+K) \times 512}

然后送进一个 Transformer encoder。

注意力 mask

建议这样：
	•	文本 token 之间全可见
	•	所有 motion token 都能看全部文本 token
	•	第 i 个 motion token 只能看历史 <= i 的 motion token

这是一个文本全可见 + motion 因果 mask。

网络

推荐：
	•	6 层 Transformer encoder
	•	8 heads
	•	FFN dim = 2048
	•	dropout = 0.1

输出 memory：

H_t \in \mathbb{R}^{B \times (8+K) \times 512}

⸻

5.3 两个解耦 decoder

然后分两支：

Plan decoder

4 个 learnable queries：

Q_c \in \mathbb{R}^{4 \times 512}

cross-attend 到 H_t，输出：

U_c \in \mathbb{R}^{B \times 4 \times 512}

再接 head：

\hat C_t = \text{Head}_c(U_c) \in \mathbb{R}^{B \times 4 \times 64}

⸻

Dynamics decoder

4 个 learnable queries：

Q_p \in \mathbb{R}^{4 \times 512}

cross-attend 到 H_t，输出：

U_p \in \mathbb{R}^{B \times 4 \times 512}

再接 head：

\hat P_t = \text{Head}_p(U_p) \in \mathbb{R}^{B \times 4 \times 38}

⸻

5.4 为什么不用一个 head 直接回归

因为你就是要让它有结构分工：
	•	plan branch 专门学 future latent pattern
	•	dynamics branch 专门学 future action evolution

这和 DreamVLA 把未来知识拆成不同类型、并且避免互相泄漏的思路是一致的。 ￼

⸻

6. 输出给 actor 的接口

actor 一般希望条件 token 维度统一。
所以需要两个 adapter：

\tilde C_t = \text{Adapter}_c(\hat C_t) \in \mathbb{R}^{B \times 4 \times d_{actor}}
\tilde P_t = \text{Adapter}_p(\hat P_t) \in \mathbb{R}^{B \times 4 \times d_{actor}}

如果你的 DiT condition dim 是 512，那就：
	•	Adapter_c: 64 -> 512
	•	Adapter_p: 38 -> 512

最终给 actor 的 cond token：

W_t = [\tilde C_t;\tilde P_t] \in \mathbb{R}^{B \times 8 \times 512}

然后 actor 接：

z_pred = actor(
    text_tokens=text_tokens_actor,     # 你原来的文本条件
    past_latents=z_past,               # 你原来的历史latent
    world_tokens=W_t                   # 新加的world cond
)

第一版建议

先不要删掉 actor 原来的 text path。
先跑通，再做更强的 bottleneck 实验。

⸻

7. target 构造细节

这是最重要的，Codex 最需要这里。

⸻

7.1 归一化

先做全局统计：

z_mean, z_std   # [64]
a_mean, a_std   # [38]

训练前统一 normalize：

z = (z - z_mean) / (z_std + 1e-6)
a = (a - a_mean) / (a_std + 1e-6)

所有 C_t^\*、P_t^\* 都在归一化空间构造。

⸻

7.2 build_c_target

def build_c_target(z_fut, Bc):
    """
    z_fut: [B, H, 64]
    Bc:    [4, H]  # fixed low-frequency DCT basis
    return: [B, 4, 64]
    """
    return torch.einsum('mh,bhd->bmd', Bc, z_fut)

DCT basis

如果 H=8，推荐用前 4 个 DCT-II basis。
可以提前离线生成，固定不训练。

⸻

7.3 build_p_target

def build_p_target(a_fut):
    """
    a_fut: [B, H, 38]
    return: [B, 4, 38]
    """
    p1 = a_fut.mean(dim=1)
    p2 = a_fut[:, -1] - a_fut[:, 0]

    da = a_fut[:, 1:] - a_fut[:, :-1]
    p3 = da.mean(dim=1)

    if a_fut.size(1) >= 3:
        d2a = da[:, 1:] - da[:, :-1]
        p4 = d2a.mean(dim=1)
    else:
        p4 = torch.zeros_like(p1)

    p = torch.stack([p1, p2, p3, p4], dim=1)
    return p


⸻

8. world model 的 loss

我建议：

\mathcal L_{wm} = \lambda_c \mathcal L_C + \lambda_p \mathcal L_P + \lambda_{aux}\mathcal L_{aux}

⸻

8.1 C_t loss

C_t 很关键，所以我建议用：
	•	SmoothL1
	•	Cosine similarity

组合：

\mathcal L_C =
\sum_{m=1}^{4} w_m
\left(
\text{SmoothL1}(\hat C_t^{(m)}, C_t^{*(m)})
+
\alpha (1-\cos(\hat C_t^{(m)}, C_t^{*(m)}))
\right)

建议权重：

[w_1,w_2,w_3,w_4] = [1.0, 0.75, 0.5, 0.25]

因为越低频的 token 越重要。

⸻

8.2 P_t loss

\mathcal L_P = \text{SmoothL1}(\hat P_t, P_t^\*)

就够了。

⸻

8.3 dynamics auxiliary head

我建议额外从 dynamics branch 再预测几个标量，方便约束它别太虚：

r_t =
[
\|a\|^2_{mean},
\|\Delta a\|^2_{mean},
\|\Delta^2 a\|^2_{mean}
]

代码：

def build_aux_target(a_fut):
    da = a_fut[:, 1:] - a_fut[:, :-1]
    if a_fut.size(1) >= 3:
        d2a = da[:, 1:] - da[:, :-1]
        jerk = (d2a ** 2).mean(dim=(1,2), keepdim=True)
    else:
        jerk = torch.zeros(a_fut.size(0), 1, device=a_fut.device)

    energy = (a_fut ** 2).mean(dim=(1,2), keepdim=True)
    vel_energy = (da ** 2).mean(dim=(1,2), keepdim=True)

    return torch.cat([energy, vel_energy, jerk], dim=1)   # [B, 3]

然后 world model 再多一个 aux_head 预测这 3 个值：

\mathcal L_{aux} = \text{MSE}(\hat r_t, r_t^\*)

⸻

8.4 推荐系数

lambda_c   = 1.0
lambda_p   = 0.5
lambda_aux = 0.1
alpha_cos  = 0.2

第一版先这么定。

⸻

9. 训练方式

这里很重要。

MotionStreamer 发现 AR latent generation 会有 exposure bias，所以用了 Two-forward；WorldVLA 也明确指出 AR action generation 会因为先前动作误差而劣化。DIAL 则采用了“先 warmup latent future，再让动作头在 GT future guidance 下学习，最后 joint”的两阶段思路。你这里最稳的训练方式也应该类似。 ￼

⸻

阶段 1：单独训练 world model

只训练：

(text,\; z_{<t},\; a_{<t}) \rightarrow (\hat C_t,\hat P_t)

监督用：

(C_t^\*, P_t^\*)

数据采样

随机采样一个序列中的时刻 t，满足：

K \le t \le T-H

然后：
	•	z_past = z[t-K:t]
	•	a_past = a[t-K:t]
	•	z_fut = z[t:t+H]
	•	a_fut = a[t:t+H]

训练细节

建议加一点 history dropout：
	•	10% 概率把部分 past latent token mask 掉
	•	10% 概率把部分 past action token mask 掉

不是为了 data augmentation 花活，而是为了让 world model 不要死记硬背最后一帧。

⸻

阶段 2：actor 用 GT world target 训练

这时 actor 接的不是 predicted world token，而是 GT：

world_tokens = concat(
    c_adapter(c_star),
    p_adapter(p_star)
)

这样 actor 先学会：

“给我未来摘要，我如何生成 next latent”

这一步非常像 DIAL 里的“System-1 在 GT future guidance 下先学执行”。 ￼

⸻

阶段 3：混合 predicted / GT

然后再逐步把 GT 换成 predicted：

use_pred = Bernoulli(prob)
world_tokens = use_pred ? pred_world : gt_world

推荐：
	•	前 20k steps：prob = 0.2
	•	中间 20k steps：prob = 0.5
	•	后面：prob = 0.8 -> 1.0

这一步本质上就是把 DIAL 的 warmup 和 MotionStreamer 的 exposure-bias 缓解思路结合一下。 ￼

⸻

10. 最推荐的超参数

这是我觉得最适合你第一版的。

K = 32          # past context
H = 8           # future horizon

M_c = 4         # number of C tokens
M_p = 4         # number of P tokens

d_model = 512
n_heads = 8
n_layers_enc = 6
n_layers_dec = 2

text_tokens = 8
dropout = 0.1

优化器：

AdamW
lr = 1e-4
weight_decay = 0.01
betas = (0.9, 0.95)

batch size 看显存。

⸻

11. 代码模块怎么拆

你可以直接让 Codex 这么建文件：

models/
    world_model.py
    world_blocks.py
    actor_adapters.py

training/
    build_world_targets.py
    losses_world.py
    train_world_model.py
    train_actor_with_world.py

configs/
    world_model.yaml


⸻

build_world_targets.py

负责：
	•	make_dct_basis(H, M_c)
	•	build_c_target(z_fut, Bc)
	•	build_p_target(a_fut)
	•	build_aux_target(a_fut)

⸻

world_model.py

提供：

class WorldModel(nn.Module):
    def forward(
        self,
        text_hidden,    # [B, Lx, Dx]
        z_past,         # [B, K, 64]
        a_past,         # [B, K, 38]
    ):
        return {
            "c_pred": c_pred,   # [B, 4, 64]
            "p_pred": p_pred,   # [B, 4, 38]
            "c_cond": c_cond,   # [B, 4, d_actor]
            "p_cond": p_cond,   # [B, 4, d_actor]
            "aux_pred": aux_pred
        }


⸻

losses_world.py

def world_loss(outputs, targets):
    # outputs: c_pred, p_pred, aux_pred
    # targets: c_star, p_star, aux_star
    ...
    return total_loss, loss_dict


⸻

12. 一个非常重要的实现建议

第一版不要做“可学习 target encoder”。

也就是：
	•	不要一开始就上一个 future latent encoder 来产出 C_t^\*
	•	不要一开始就上一个 future action encoder 来产出 P_t^\*

因为那样会立刻引入：
	•	target 漂移
	•	teacher collapse
	•	训练不稳
	•	debug 困难

你现在最需要的是一个确定性的、可解释的、可复现实验。
所以：
	•	C_t^\*：固定 DCT 低频基压 future latent
	•	P_t^\*：固定统计量压 future action

这版最适合先跑起来。

等第一版结果稳定了，再升级到：
	•	learnable plan target encoder
	•	learnable dynamics target encoder
	•	更强的 joint training

⸻

13. 你这版 world model 的一句话定义

你可以把它写成：

\text{WM}: (x, z_{t-K:t-1}, a_{t-K:t-1})
\rightarrow
(\hat C_t, \hat P_t)

其中：

C_t^\* = \text{LowFreqDCT}\big(z_{t:t+H-1}\big)
P_t^\* = \text{DynStats}\big(a_{t:t+H-1}\big)

也就是：
	•	C_t：未来 latent 块的低频计划摘要
	•	P_t：未来 action 块的动态统计摘要

这两个一起喂给 actor 去预测下一步 latent。

⸻

14. 我对你这个版本的最终建议

如果你现在就要给 Codex，我建议你直接下这条指令思路：
	1.	先把数据整理成 [B,T,38] 和 [B,T,64]
	2.	实现 deterministic C_target 和 P_target
	3.	单独训练 WorldModel
	4.	让 actor 先吃 GT world token
	5.	再切 predicted world token

这是最稳的路径。

真正最关键的不是花哨，而是：

先把 C_t 和 P_t 设计成“来源清楚、监督清楚、代码清楚”的两个对象。

你现在这版已经够清楚了。

下一条我可以直接给你一份 可以贴给 Codex 的英文 implementation spec，包括类名、函数签名、shape 注释和训练循环。