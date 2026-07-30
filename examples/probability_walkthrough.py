"""用可手算的数字解释 VAE 所需的概率、期望、MLE 和 ELBO。

只依赖 Python 标准库：
    python examples/probability_walkthrough.py
"""

from __future__ import annotations

import math


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def discrete_latent_example() -> None:
    """计算联合、边缘、后验和期望。"""
    section("1. 联合、边缘、后验：天气 z -> 延误 x")

    p_z = {0: 0.6, 1: 0.4}
    p_x1_given_z = {0: 0.2, 1: 0.9}

    joint = {
        z: p_z[z] * p_x1_given_z[z]
        for z in p_z
    }
    p_x1 = sum(joint.values())
    posterior = {
        z: joint[z] / p_x1
        for z in p_z
    }

    for z in p_z:
        print(
            f"p(x=1,z={z}) = p(z={z}) p(x=1|z={z})"
            f" = {p_z[z]:.2f} * {p_x1_given_z[z]:.2f}"
            f" = {joint[z]:.2f}"
        )
    print(f"p(x=1) = sum_z p(x=1,z) = {p_x1:.2f}")
    print(
        "p(z=1|x=1) = p(x=1,z=1) / p(x=1)"
        f" = {joint[1]:.2f} / {p_x1:.2f} = {posterior[1]:.2f}"
    )

    f = {0: 10.0, 1: 30.0}
    prior_expectation = sum(p_z[z] * f[z] for z in p_z)
    posterior_expectation = sum(posterior[z] * f[z] for z in p_z)
    print("\n同一个 f(z)，期望下标不同，权重和结果就不同：")
    print(f"E_p(z)[f(z)]     = {prior_expectation:.2f}")
    print(f"E_p(z|x=1)[f(z)] = {posterior_expectation:.2f}")


def maximum_likelihood_example() -> None:
    """通过网格搜索说明为什么最大化观测数据的似然。"""
    section("2. 最大似然：选择最能解释三正一反的参数 theta")

    observations = [1, 1, 1, 0]
    candidates = [i / 100 for i in range(1, 100)]

    def log_likelihood(theta: float) -> float:
        return sum(
            x * math.log(theta) + (1 - x) * math.log(1 - theta)
            for x in observations
        )

    best_theta = max(candidates, key=log_likelihood)
    for theta in [0.10, 0.50, 0.75, 0.90]:
        likelihood = theta ** sum(observations) * (
            1 - theta
        ) ** (len(observations) - sum(observations))
        print(
            f"theta={theta:.2f}: "
            f"p_theta(D)={likelihood:.8f}, "
            f"log p_theta(D)={log_likelihood(theta):.6f}"
        )

    print(f"\n0.01 网格上的最大似然参数：theta={best_theta:.2f}")
    print("解析解是样本均值：(1+1+1+0)/4 = 0.75")


def kl_bernoulli(q1: float, p1: float) -> float:
    """KL(Bernoulli(q1) || Bernoulli(p1))."""
    return (
        q1 * math.log(q1 / p1)
        + (1 - q1) * math.log((1 - q1) / (1 - p1))
    )


def exact_elbo_example() -> None:
    """在离散模型中验证 log p(x) = ELBO + posterior KL。"""
    section("3. ELBO：精确验证“下界 + 缺口 = 对数证据”")

    p_z1 = 0.4
    p_x1_given_z = {0: 0.2, 1: 0.9}
    p_x1 = (1 - p_z1) * p_x1_given_z[0] + p_z1 * p_x1_given_z[1]
    true_posterior_z1 = p_z1 * p_x1_given_z[1] / p_x1
    log_evidence = math.log(p_x1)

    print(f"真实 log p(x=1) = log({p_x1:.2f}) = {log_evidence:.6f}\n")
    print(" q(z=1|x)       ELBO       KL(q||posterior)     ELBO+KL")

    for q_z1 in [0.10, 0.40, 0.75, 0.90]:
        q = {0: 1 - q_z1, 1: q_z1}
        expected_log_likelihood = sum(
            q[z] * math.log(p_x1_given_z[z])
            for z in [0, 1]
        )
        kl_to_prior = kl_bernoulli(q_z1, p_z1)
        elbo = expected_log_likelihood - kl_to_prior
        gap = kl_bernoulli(q_z1, true_posterior_z1)
        print(
            f"    {q_z1:0.2f}       {elbo: .6f}"
            f"         {gap: .6f}          {elbo + gap: .6f}"
        )

    print(
        "\n当 q(z=1|x)=0.75 等于真实后验时，KL 缺口为 0，"
        "ELBO 恰好等于 log p(x)。"
    )


def main() -> None:
    discrete_latent_example()
    maximum_likelihood_example()
    exact_elbo_example()


if __name__ == "__main__":
    main()
