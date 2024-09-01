# 线性代数

> 转置：
>
> ```python
> A.T
> ```
>
> 两个矩阵的按元素乘法称为Hadamard积（Hadamard product）

## 降维

### 计算其元素的和

```python
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

### 指定张量沿哪一个轴来通过求和降低维度

```python
A_sum_axis0 = A.sum(axis=0)
A, A_sum_axis0, A_sum_axis0.shape
```

out:

```text
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 tensor([ 6., 22., 38., 54., 70.]),
 torch.Size([5]))
```

沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和:

```python
A.sum(axis=[0, 1])  # 结果和A.sum()相同
```

### 平均值

```python
A.mean(), A.sum() / A.numel()

# 沿指定轴降低张量的维度
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

## 非降维求和

### `keepdims=True`: 计算总和或均值时保持轴数不变

```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

out:

```text
tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])
```

> 例如，由于sum_A在对每行进行求和后仍保持两个轴，我们可以通过广播将A除以sum_A。

### `cumsum`: 沿某个轴计算A元素的累积总和

```python
A.cumsum(axis=0)
```

## 点积（Dot Product）`dot`

> 点积是相同位置的按元素乘积的和

```python
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)

# 可以通过执行按元素乘法，然后进行求和来表示两个向量的点积
torch.sum(x * y)
```

## 矩阵-向量积 `mv`

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

> 为矩阵A和向量x调用torch.mv(A, x)时，会执行矩阵-向量积。 注意，A的列维数（沿轴1的长度）必须与x的维数（其长度）相同。

```python
A.shape, x.shape, torch.mv(A, x)
```

## 矩阵-矩阵乘法 `mm`

```python
B = torch.ones(4, 3)
torch.mm(A, B)
```

> 矩阵-矩阵乘法可以简单地称为**矩阵乘法**，不应与"Hadamard积"混淆。

## 范数

线性代数中最有用的一些运算符是*范数*（norm）。
非正式地说，向量的*范数*是表示一个向量有多大。
这里考虑的*大小*（size）概念不涉及维度，而是分量的大小。

在线性代数中，向量范数是将向量映射到标量的函数$f$。
给定任意向量$\mathbf{x}$，向量范数要满足一些属性。
第一个性质是：如果我们按常数因子$\alpha$缩放向量的所有元素，
其范数也会按相同常数因子的*绝对值*缩放：

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

第二个性质是熟悉的三角不等式:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

第三个性质简单地说范数必须是非负的:

$$f(\mathbf{x}) \geq 0.$$

这是有道理的。因为在大多数情况下，任何东西的最小的*大小*是0。
最后一个性质要求范数最小为0，当且仅当向量全由0组成。

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

范数听起来很像距离的度量。
欧几里得距离和毕达哥拉斯定理中的非负性概念和三角不等式可能会给出一些启发。
事实上，欧几里得距离是一个$L_2$范数：
假设$n$维向量$\mathbf{x}$中的元素是$x_1,\ldots,x_n$，其[**$L_2$*范数*是向量元素平方和的平方根：**]

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$**)

其中，在$L_2$范数中常常省略下标$2$，也就是说$\|\mathbf{x}\|$等同于$\|\mathbf{x}\|_2$。
在代码中，我们可以按如下方式计算向量的$L_2$范数。

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```
深度学习中更经常地使用$L_2$范数的平方，也会经常遇到[**$L_1$范数，它表示为向量元素的绝对值之和：**]

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

与$L_2$范数相比，$L_1$范数受异常值的影响较小。
为了计算$L_1$范数，我们将绝对值函数和按元素求和组合起来。

```python
torch.abs(u).sum()
```

$L_2$范数和$L_1$范数都是更一般的$L_p$范数的特例：

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

类似于向量的$L_2$范数，[**矩阵**]$\mathbf{X} \in \mathbb{R}^{m \times n}$(**的*Frobenius范数*（Frobenius norm）是矩阵元素平方和的平方根：**)

(**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**)

Frobenius范数满足向量范数的所有性质，它就像是矩阵形向量的$L_2$范数。
调用以下函数将计算矩阵的Frobenius范数。

```python
torch.norm(torch.ones((4, 9)))
```

> 在深度学习中，我们经常试图解决优化问题： 最大化分配给观测数据的概率; 最小化预测和真实观测之间的距离。 用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。 目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。

