"""逐个打印一次 VAE 损失计算中的张量，不下载数据也不训练网络。

运行：
    python examples/vae_loss_walkthrough.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(7)

    # 两个“样本”，每个样本只有 4 个二值特征，便于直接观察数字。
    x = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
    )

    # 把它们想成 encoder(x) 的输出。logvar = log(sigma^2)。
    mu = torch.tensor([[0.20, -0.10], [0.70, 0.30]], requires_grad=True)
    logvar = torch.tensor(
        [[0.00, -0.40], [0.20, 0.60]], requires_grad=True
    )
    std = torch.exp(0.5 * logvar)

    # epsilon 来自固定标准正态；z 对 mu 和 std 保持可微。
    epsilon = torch.randn_like(std)
    z = mu + std * epsilon

    # 假设这些 logits 是 decoder(z) 的输出。
    # 本演示只关注损失各项，所以不构造完整解码器。
    logits = torch.tensor(
        [[1.30, -0.80, 0.60, -0.20], [-0.40, 1.10, 0.20, -1.20]],
        requires_grad=True,
    )

    # reduction="none" 保留每个元素，再按样本求和。
    reconstruction_nll = F.binary_cross_entropy_with_logits(
        logits, x, reduction="none"
    ).sum(dim=1)

    # KL(N(mu, diag(sigma^2)) || N(0, I)) 的闭式解。
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

    negative_elbo_per_sample = reconstruction_nll + kl
    loss = negative_elbo_per_sample.mean()
    loss.backward()

    print("x =\n", x)
    print("\nmu =\n", mu.detach())
    print("\nlogvar = log(sigma^2) =\n", logvar.detach())
    print("\nstd = exp(0.5 * logvar) =\n", std.detach())
    print("\nepsilon ~ N(0,I) =\n", epsilon)
    print("\nz = mu + std * epsilon =\n", z.detach())
    print("\n每个样本的重构负对数似然 =", reconstruction_nll.detach())
    print("每个样本的 KL(q(z|x)||p(z)) =", kl.detach())
    print("每个样本的负 ELBO =", negative_elbo_per_sample.detach())
    print("batch 平均 loss =", loss.item())
    print("\n反向传播后：")
    print("d loss / d mu =\n", mu.grad)
    print("d loss / d logvar =\n", logvar.grad)
    print("d loss / d logits =\n", logits.grad)
    print(
        "\n结论：代码最小化 reconstruction_nll + KL，"
        "等价于最大化 ELBO。"
    )


if __name__ == "__main__":
    main()
