# 公式与代码对应表

核心代码位于 `src/vae_lab/models.py`。

| 数学对象 | 代码对象 | 实现要点 |
|---|---|---|
| $\mu_\phi(x),\log\sigma_\phi^2(x)$ | `GaussianVAE.encode` | 一个隐藏表示分成两个线性头 |
| $z=\mu+\sigma\odot\epsilon$ | `reparameterize` | `exp(0.5 * logvar)` |
| $D_{KL}(q\Vert p)$ | `kl_standard_normal` | 对潜变量维求和，对 batch 保留 |
| $-\log p_\theta(x\mid z)$ | `bernoulli_nll` | 对 logits 使用稳定 BCE |
| $-\mathcal L_\beta$ | `vae_loss` | `reconstruction + beta * kl` |
| $q(z\mid x,c)$ | `ConditionalVAE.encode` | 拼接 one-hot 条件 |
| $p(x\mid z,c)$ | `ConditionalVAE.decode` | 解码时再次拼接条件 |
| $\log p(x,z)-\log q(z\mid x)$ | `iwae_loss` | 显式计算三个 log density |
| $\log\sum\exp-\log K$ | `iwae_loss` | `torch.logsumexp` |
| $\arg\min_k\|z_e-e_k\|^2$ | `VectorQuantizer.forward` | 展开平方距离并 `argmin` |
| $\operatorname{sg}$ | `.detach()` | 前向保值、反向截断 |
| $z_e+\operatorname{sg}[z_q-z_e]$ | `quantized_st` | 直通估计 |

## 张量形状

### GaussianVAE

```text
x             [B, 1, 28, 28]
flatten(x)    [B, 784]
mu/logvar     [B, latent_dim]
z             [B, latent_dim]
logits        [B, 1, 28, 28]
```

### IWAE

```text
mu/logvar     [B, D]
epsilon       [K, B, D]
z             [K, B, D]
logits        [K, B, 1, 28, 28]
log_weight    [K, B]
loss          scalar
```

### VQ‑VAE

```text
z_e           [B, D, H, W]
flat_z        [B*H*W, D]
distances     [B*H*W, K]
indices       [B, H, W]
z_q           [B, D, H, W]
```

## 数值稳定性检查

- 解码器输出 logits，不在模型中做 sigmoid；
- Bernoulli NLL 使用 `binary_cross_entropy_with_logits`；
- IWAE 聚合使用 `logsumexp`，不先对大数取指数；
- 对数方差在编码器中适度裁剪，避免 `exp(logvar)` 溢出；
- VQ 距离直接用平方距离展开，避免构造超大的广播差值张量。

