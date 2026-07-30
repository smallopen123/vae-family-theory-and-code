# 论文与参考代码

## 原始论文与理论资料

1. Kingma, D. P., & Welling, M. (2013). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). VAE、重参数化估计器与 SGVB 的原始论文。
2. Kingma, D. P., & Welling, M. (2019). [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691). 更系统的 VAE 教程。
3. Sohn, K., Lee, H., & Yan, X. (2015). [Learning Structured Output Representation using Deep Conditional Generative Models](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html). CVAE 代表性论文。
4. Higgins, I. et al. (2017). [β‑VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl).
5. Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519).
6. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). VQ‑VAE 原始论文。

## 概率基础学习路线

1. Kevin P. Murphy. [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html). MIT Press, 2022。官方页面提供合法公开草稿、目录、图和配套代码；适合补随机变量、概率密度、条件概率、期望、最大似然和潜变量模型。
2. [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) 比原始 VAE 论文更适合作为第二遍系统阅读。建议先读本仓库的数字例子，再读其中的潜变量模型、变分推断和重参数化章节。
3. [PyTorch 官方 Examples 入口](https://docs.pytorch.org/examples/) 与其 [VAE 源代码](https://github.com/pytorch/examples/tree/main/vae) 适合核对 `BCE + KL`、重参数化和训练循环；短代码示例不能替代概率推导。

## 经检索的参考实现

下列仓库用于交叉核对常见工程写法。本仓库代码是统一接口下的独立教学实现，并非复制：

1. [PyTorch 官方 examples/vae](https://github.com/pytorch/examples/tree/main/vae)：基础 VAE 的简洁基线。
2. [Keras 官方 VAE 示例](https://keras.io/examples/generative/vae/)：卷积 VAE、采样层与可视化。
3. [1Konny/Beta-VAE](https://github.com/1Konny/Beta-VAE)：β‑VAE 公开实现。
4. [zalandoresearch/pytorch-vq-vae](https://github.com/zalandoresearch/pytorch-vq-vae)：PyTorch VQ‑VAE 实现。
5. [Keras 官方 VQ‑VAE 示例](https://keras.io/examples/generative/vq_vae/)：码本量化与离散先验示例。
6. [leimao/PyTorch-Variational-Autoencoder](https://github.com/leimao/PyTorch-Variational-Autoencoder)：完整 PyTorch VAE 工程参考。

## 阅读时的核查建议

- 先确认实现用的是 Bernoulli、Gaussian 还是其他似然；
- 比较 loss 的 `sum` / `mean` 约定后再比较数值；
- 检查 `logvar` 是否真的是 $\log\sigma^2$；
- VQ‑VAE 需确认码本使用梯度、EMA 还是两者之一更新；
- 代码能运行不等于公式假设一致，实验比较前应统一数据预处理和似然。

