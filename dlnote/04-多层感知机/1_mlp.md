# 多层感知机

## 隐藏层

> 线性模型不能解决XOR问题，难以通过简单的线性模型解决一些复杂的问题。可以引入一个或多个隐藏层来克服线性模型的限制。

>设通过矩阵$\mathbf{X} \in \mathbb{R}^{n \times d}$来表示$n$个样本的小批量，其中每个样本具有$d$个输入特征。

对于具有$h$个隐藏单元的单隐藏层多层感知机，用$\mathbf{H} \in \mathbb{R}^{n \times h}$表示隐藏层的输出，称为*隐藏表示*（hidden representations）。

在数学或代码中，$\mathbf{H}$也被称为*隐藏层变量*（hidden-layer variable）或*隐藏变量*（hidden variable）。

因为隐藏层和输出层都是全连接的，
所以我们有隐藏层权重$\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$
和隐藏层偏置$\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$
以及输出层权重$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$
和输出层偏置$\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$。

为了发挥多层架构的潜力，
我们还需要一个额外的关键要素：
在仿射变换之后对每个隐藏单元应用非线性的*激活函数*（activation function）$\sigma$。
激活函数的输出被称为*活性值*（activations）。

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

由于$\mathbf{X}$中的每一行对应于小批量中的一个样本，
出于记号习惯的考量，
定义非线性函数$\sigma$也以按行的方式作用于其输入，
即一次计算一个样本。

> **若无激活函数，则多层感知机将退化为线性模型**

## 激活函数

### 1. ReLU函数

最受欢迎的激活函数是***修正线性单元***（Rectified linear unit，*ReLU*），
因为它实现简单，同时在各种预测任务中表现良好。
**ReLU提供了一种非常简单的非线性变换**。
给定元素$x$，ReLU函数被定义为该元素与$0$的最大值：

**$$\operatorname{ReLU}(x) = \max(x, 0).$$**

当输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1。
当输入值精确等于0时，ReLU函数不可导。此时默认使用左侧的导数，即当输入为0时导数为0。

### 2. sigmoid函数

**对于一个定义域在$\mathbb{R}$中的输入，
*sigmoid函数*将输入变换为区间(0, 1)上的输出**。
因此，sigmoid通常称为*挤压函数*（squashing function）：
它将范围（-inf, inf）中的任意输入压缩到区间（0, 1）中的某个值：

**$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$**

sigmoid函数是一个平滑的、可微的阈值单元近似。 然而，sigmoid在隐藏层中已经较少使用， 它在大部分时候被更简单、更容易训练的ReLU所取代。

### 3. tanh函数

与sigmoid函数类似，
**tanh(双曲正切)函数也能将其输入压缩转换到区间(-1, 1)上**。
tanh函数的公式如下：

**$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$**
