# 看懂 VAE 前必须会的概率语言

> 目标：读完本章后，你应该能逐字解释  
> $`p_\theta(x)`$、$`p_\theta(x,z)`$、$`p_\theta(x\mid z)`$、$`p_\theta(z\mid x)`$、
> $`q_\phi(z\mid x)`$ 和 $`\mathbb E_{q_\phi(z\mid x)}[\cdot]`$，并能回答
> “为什么训练要最大化 $`p_\theta(x)`$”。
>
> 本章公式使用 GitHub 原生 LaTeX 数学块，网页打开即可查看；同时保留
> 可复制、可修改的 LaTeX 源码，不需要转换或安装插件。

## 0. 先记住一张“符号身份证”

| 符号 | 它是什么 | 一句话解释 |
|---|---|---|
| $`x`$ | 已观察到的数据 | 例如当前这张手写数字图片 |
| $`z`$ | 未观察到的潜变量 | 例如字形、倾斜度、笔画粗细等生成因素 |
| $`\theta`$ | 生成模型参数 | 解码器中的权重；决定模型怎样由 $`z`$ 生成 $`x`$ |
| $`\phi`$ | 推断模型参数 | 编码器中的权重；决定看见 $`x`$ 后怎样猜测 $`z`$ |
| $`p_\theta`$ | 模型声明的分布 | 下标 $`\theta`$ 表示这个分布会随生成参数而改变 |
| $`q_\phi`$ | 近似后验分布 | 用可计算的编码器近似难算的真实后验 |

最重要的阅读习惯是：

1. 圆括号里写“正在讨论哪个随机量”；
2. 条件线 $`\mid`$ 右边写“已经知道什么”；
3. 分布字母下面的角标写“用哪个分布做加权或采样”；
4. $`p`$ 和 $`q`$ 不是两个固定公式，而是两套分布的名字。

---

## 1. $`p(x)`$ 不是“这张图片本身的概率”

### 1.1 离散变量：概率可以直接相加

若 $`X`$ 表示一次抛硬币的结果，用 $`H`$ 表示正面、$`T`$ 表示反面，那么

```math
p(X=H)=0.6,\qquad p(X=T)=0.4.
\qquad \text{(P1)}
```

通常为了简洁，把随机变量 $`X`$ 取到具体值 $`x`$ 的概率写成 $`p(x)`$。
所以 $`p(x)`$ 的准确读法是：

> “随机变量 $`X`$ 取到值 $`x`$ 时，模型分配给它的概率质量。”

### 1.2 连续变量：$`p(x)`$ 通常是密度，不是单点概率

图像像素或高斯潜变量一般按连续量处理。连续变量在单个点的概率为零，
但一个很小区间的概率近似等于“密度乘区间宽度”：

```math
\Pr(x\le X\le x+\Delta x)\approx p(x)\,\Delta x.
\qquad \text{(P2)}
```

所以 VAE 中口头说“提高图片 $`x`$ 的概率”，更严谨地说是：

> 提高模型在观测数据附近分配的概率密度。

密度可以大于 1；真正必须等于 1 的是整个空间下的积分：

```math
\int p(x)\,dx=1.
\qquad \text{(P3)}
```

---

## 2. 联合、条件、边缘：三个符号到底差在哪里

先使用一个可以手算完的“天气—航班延误”潜变量模型：

- $`Z=0`$：天气正常，$`p(z=0)=0.6`$；
- $`Z=1`$：天气恶劣，$`p(z=1)=0.4`$；
- $`X=1`$：航班延误；
- $`p(x=1\mid z=0)=0.2`$；
- $`p(x=1\mid z=1)=0.9`$。

### 2.1 条件概率 $`p(x\mid z)`$

$`p(x=1\mid z=0)=0.2`$ 的读法是：

> 已经知道天气正常时，航班延误的概率为 0.2。

条件线不是除法，也不是模型的输入箭头；它表达“在已知右侧的前提下，
左侧变量服从什么分布”。

### 2.2 联合概率 $`p(x,z)`$

“天气恶劣并且航班延误”的联合概率为

```math
p(x=1,z=1)=p(z=1)\,p(x=1\mid z=1)=0.4\times0.9=0.36.
\qquad \text{(P4)}
```

一般地，联合分布的乘法规则是

```math
p(x,z)=p(z)\,p(x\mid z)=p(x)\,p(z\mid x).
\qquad \text{(P5)}
```

VAE 选择第一种分解作为生成故事：先生成原因 $`z`$，再由原因生成观测 $`x`$。

### 2.3 边缘概率 $`p(x)`$：把不知道的 $`z`$ 加起来

我们只看见“延误”，不知道天气状态。因此要把所有可能的天气状态都考虑进去：

```math
\begin{aligned}
p(x=1)
&=\sum_z p(x=1,z)\\
&=p(z=0)p(x=1\mid z=0)+p(z=1)p(x=1\mid z=1)\\
&=0.6\times0.2+0.4\times0.9\\
&=0.48.
\end{aligned}
\qquad \text{(P6)}
```

这个“把隐藏原因全部加掉”的动作叫**边缘化**。若 $`z`$ 连续，就把求和换成积分：

```math
p_\theta(x)=\int p_\theta(x,z)\,dz
=\int p(z)\,p_\theta(x\mid z)\,dz.
\qquad \text{(P7)}
```

注意：这里不是“积分掉了无用信息”，而是说我们没有观测到 $`z`$，所以必须把每个
可能的 $`z`$ 生成当前 $`x`$ 的贡献都累计起来。

### 2.4 后验概率 $`p(z\mid x)`$：看见结果后反推原因

现在已经看见航班延误，天气恶劣的概率是多少？根据 Bayes 公式：

```math
\begin{aligned}
p(z=1\mid x=1)
&=\frac{p(x=1,z=1)}{p(x=1)}\\
&=\frac{p(z=1)p(x=1\mid z=1)}{p(x=1)}\\
&=\frac{0.4\times0.9}{0.48}=0.75.
\end{aligned}
\qquad \text{(P8)}
```

先验 $`p(z=1)=0.4`$ 是“没看到延误前”的判断；后验 $`p(z=1\mid x=1)=0.75`$
是“看到延误后”的更新判断。

VAE 中也是同一件事：

- $`p(z)`$：看图之前，潜变量可能在哪里；
- $`p_\theta(z\mid x)`$：看见图片后，哪些潜变量最可能生成它；
- $`q_\phi(z\mid x)`$：编码器对这个难算后验的近似。

---

## 3. 期望符号的下标究竟表示什么

### 3.1 期望就是“按概率加权的平均”

设函数 $`f(z)`$ 在 $`z=0`$ 时取 10，在 $`z=1`$ 时取 30。按先验 $`p(z)`$ 加权：

```math
\mathbb E_{p(z)}[f(z)]
=\sum_z p(z)f(z)
=0.6\times10+0.4\times30=18.
\qquad \text{(P9)}
```

下标 $`p(z)`$ 回答的不是“对谁求导”，而是两个问题：

1. 用谁的概率作为权重？
2. 如果用采样近似，样本从哪个分布抽？

若改用看见延误后的后验，权重发生变化：

```math
\mathbb E_{p(z\mid x=1)}[f(z)]
=0.25\times10+0.75\times30=25.
\qquad \text{(P10)}
```

因此，即使方括号中的 $`f(z)`$ 完全相同，期望下标不同，结果也会不同。

### 3.2 连续变量时，求和变成积分

```math
\mathbb E_{q_\phi(z\mid x)}[f(z)]
=\int q_\phi(z\mid x)\,f(z)\,dz.
\qquad \text{(P11)}
```

逐字读作：

> 固定当前观测 $`x`$，让 $`z`$ 按编码器给出的
> $`q_\phi(z\mid x)`$ 分布变化，对 $`f(z)`$ 做概率加权平均。

### 3.3 代码里的样本平均是在近似期望

积分难以直接计算时，从该分布采样 $`L`$ 次：

```math
\mathbb E_{q_\phi(z\mid x)}[f(z)]
\approx \frac1L\sum_{\ell=1}^{L}f(z^{(\ell)}),
\qquad z^{(\ell)}\sim q_\phi(z\mid x).
\qquad \text{(P12)}
```

这里：

- $`\sim`$ 读作“服从”或“从……采样”；
- 上标 $`(\ell)`$ 是第 $`\ell`$ 个样本，不是幂；
- 样本越多，Monte Carlo 平均通常越稳定；
- 训练 VAE 时常令 $`L=1`$，依靠许多批次累计得到有效梯度。

---

## 4. 为什么要最大化 $`p_\theta(x)`$

### 4.1 参数 $`\theta`$ 改变，模型对数据的评价就改变

先看最简单的 Bernoulli 模型。$`\theta`$ 表示正面概率。观测数据为

```math
\mathcal D=(1,1,1,0).
\qquad \text{(P13)}
```

假设四次观测条件独立，整组数据在参数 $`\theta`$ 下的似然为

```math
p_\theta(\mathcal D)
=\prod_{i=1}^{4}p_\theta(x_i)
=\theta^3(1-\theta).
\qquad \text{(P14)}
```

它不是“参数 $`\theta`$ 的概率”；$`\theta`$ 在最大似然中是我们可以调整的旋钮。
似然回答：

> 如果模型参数取这个值，已经真实发生的这组数据有多合理？

比较三个候选参数：

| $`\theta`$ | $`p_\theta(\mathcal D)=\theta^3(1-\theta)`$ |
|---:|---:|
| 0.10 | 0.0009 |
| 0.50 | 0.0625 |
| 0.75 | 0.10546875 |

$`\theta=0.75`$ 给已经观察到的“三正一反”最高的似然。最大化似然就是选择
最能解释训练样本的模型参数：

```math
\theta_{\mathrm{MLE}}
=\arg\max_\theta p_\theta(\mathcal D).
\qquad \text{(P15)}
```

### 4.2 为什么实际最大化对数似然

对数函数严格单调递增，所以最大值位置不变：

```math
\arg\max_\theta p_\theta(\mathcal D)
=\arg\max_\theta\log p_\theta(\mathcal D).
\qquad \text{(P16)}
```

对数还能把许多很小概率的乘积变为求和：

```math
\log p_\theta(\mathcal D)
=\sum_{i=1}^{N}\log p_\theta(x_i).
\qquad \text{(P17)}
```

这样既便于小批量训练，又避免大量小数相乘造成数值下溢。

### 4.3 放回 VAE：我们最大化的是观测数据的边缘似然

每张训练图片只给出 $`x_i`$，没有给出对应 $`z_i`$。因此需要最大化

```math
\sum_{i=1}^{N}\log p_\theta(x_i)
=\sum_{i=1}^{N}\log\int p(z_i)p_\theta(x_i\mid z_i)\,dz_i.
\qquad \text{(P18)}
```

直观上，它要求解码器和先验共同做到：

- 对训练集中真实出现的图片分配较高密度；
- 存在足够多的合理潜变量区域能够生成这些图片；
- 不只是某个编码结果能重构，而是整个概率生成模型能解释数据。

“最大化 $`p_\theta(x)`$”不是让像素数值变大，也不是让网络输出趋近 1。
它是在所有可能参数中，让真实数据在模型分布下尽可能合理。

---

## 5. 为什么不能直接算 $`p_\theta(x)`$，又为什么引入 $`q_\phi`$

VAE 的后验由 Bayes 公式给出：

```math
p_\theta(z\mid x)
=\frac{p(z)p_\theta(x\mid z)}
{\int p(z')p_\theta(x\mid z')\,dz'}.
\qquad \text{(P19)}
```

分母需要遍历连续高维潜空间。解码器是非线性神经网络时，这个积分通常没有
可用的解析解。注意，“后验难算”和“后验不存在”是两回事：它在数学上存在，
只是计算代价不可接受。

于是用编码器输出一个容易采样、容易算密度的分布：

```math
q_\phi(z\mid x)
=\mathcal N\!\left(z;\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))\right).
\qquad \text{(P20)}
```

$`q`$ 的角色不是替换生成模型中的先验 $`p(z)`$，而是帮助我们在看见 $`x`$ 后，
快速找到最可能解释它的潜变量区域。

---

## 6. ELBO：把一个难算目标拆成两个可训练目标

有一个精确恒等式：

```math
\log p_\theta(x)
=\mathcal L(\theta,\phi;x)
+D_{\mathrm{KL}}\!\left(
q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right).
\qquad \text{(P21)}
```

KL 散度永远非负，所以

```math
\log p_\theta(x)\ge \mathcal L(\theta,\phi;x).
\qquad \text{(P22)}
```

这就是“证据下界”：证据指 $`p_\theta(x)`$，下界指 $`\mathcal L`$ 不超过它。
这个恒等式不是需要死记的结论；[基础 VAE 推导第 3、4 节](01-vae-derivation.md)
分别从 Jensen 不等式和 KL 非负性出发，把乘除 $`q_\phi`$、展开对数、移项的
每一行都写了出来，并全部使用网页可直接查看的公式图片。建议先理解本章的
数字 ELBO 实验，再沿着那两条推导各走一遍。

把联合分布 $`p_\theta(x,z)=p(z)p_\theta(x\mid z)`$ 代入，下界可写成

```math
\mathcal L(\theta,\phi;x)
=\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right).
\qquad \text{(P23)}
```

逐项解释：

1. **重构期望**：从编码器认为合理的 $`z`$ 区域采样，解码器应给当前 $`x`$
   较高的条件对数似然；
2. **KL 正则**：编码器给出的每个后验不能随意漂离统一先验，否则从先验采样时
   会落入解码器没见过的空洞区域；
3. 最大化“重构项减 KL”，等价于最小化“负重构项加 KL”。

因此 PyTorch 中常见的损失是

```math
\mathrm{loss}
=-\mathcal L
=\underbrace{-\mathbb E_q[\log p_\theta(x\mid z)]}_{\text{重构负对数似然}}
+\underbrace{D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))}_{\text{KL 正则}}.
\qquad \text{(P24)}
```

若像素采用 Bernoulli 似然，第一项就是二元交叉熵 BCE。代码里的
`BCE + KL` 并不是另造了一个目标，而是负 ELBO。

---

## 7. 从公式到一次训练的完整信息流

```mermaid
flowchart LR
    X["观测图片 x"] --> E["编码器参数 φ"]
    E --> Q["qφ(z|x) 的 μ 与 log σ²"]
    Q --> S["采样 z"]
    S --> D["解码器参数 θ"]
    D --> PX["pθ(x|z) 的参数"]
    PX --> R["重构负对数似然"]
    Q --> K["KL(qφ(z|x) || p(z))"]
    R --> LOSS["loss = 重构 NLL + KL"]
    K --> LOSS
    LOSS --> B["反向传播，同时更新 θ 与 φ"]
```

重参数化把“从依赖 $`\phi`$ 的分布采样”改写为

```math
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
\qquad \text{(P25)}
```

随机性被隔离在不含参数的 $`\epsilon`$ 中，$`z`$ 对 $`\mu_\phi`$ 和
$`\sigma_\phi`$ 仍是可微的，因此重构项的梯度可以传回编码器。

---

## 8. 运行顺序：不要一开始就训练 MNIST

### 第一步：只用 Python 标准库，亲手核对所有概率

```bash
python examples/probability_walkthrough.py
```

它会打印：

- 联合概率、边缘概率、Bayes 后验；
- 两个不同分布下的期望；
- Bernoulli 最大似然的网格搜索；
- 一个可以精确计算的 ELBO 及其 KL 缺口。

运行时把本章式 (P4) 到 (P23) 放在旁边，逐行核对数字。

### 第二步：用 PyTorch 看负 ELBO 的每个张量

```bash
python examples/vae_loss_walkthrough.py
```

脚本不下载数据、不训练网络，只构造两个样本，打印 `mu`、`logvar`、
`std`、`epsilon`、`z`、每个样本的重构 NLL、KL 和总损失。先弄清每个张量，
再运行完整 MNIST 训练。

### 第三步：运行单元测试和一分钟短训练

```bash
pytest -q
python -m vae_lab.train --model vae --epochs 1
```

首次训练会下载 MNIST。理解基础 VAE 后，再按
[变体推导](02-variants.md) 的顺序学习 $`\beta`$-VAE、CVAE、IWAE 和 VQ-VAE。

---

## 9. 自测：能说清这些问题再进入下一章

1. $`p_\theta(x\mid z)`$ 与 $`p_\theta(z\mid x)`$ 的条件方向为何不同？
2. 为什么 $`p_\theta(x)`$ 中看不到 $`z`$，但计算它仍必须考虑所有 $`z`$？
3. $`\mathbb E_{q_\phi(z\mid x)}[f(z)]`$ 的下标具体决定什么？
4. 为什么最大化似然不是让像素值变大？
5. 为什么对数不会改变最大似然解？
6. $`q_\phi(z\mid x)`$ 是近似哪个分布，为什么需要近似？
7. `BCE + KL` 为什么等于最小化负 ELBO？
8. 重参数化中真正被随机采样的是哪个变量？

如果其中任何一题仍含糊，请回到对应的数字例子，而不是继续背 ELBO。

## 10. 推荐资料及使用方式

- Kingma 与 Welling，*Auto-Encoding Variational Bayes*：VAE 原始论文；
  先读算法图与重参数化，再读附录推导。
- Kingma 与 Welling，*An Introduction to Variational Autoencoders*：
  比原始论文更适合作为第二遍系统阅读。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*：
  用于补概率、期望、最大似然和潜变量模型基础。
- PyTorch 官方 `examples/vae`：用于核对 BCE、KL、重参数化和训练循环。

准确链接和章节定位见[参考资料](04-references.md)。
