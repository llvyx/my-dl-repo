# 自动微分

> 求导是几乎所有深度学习优化算法的关键步骤。深度学习框架通过自动计算导数，即自动微分（automatic differentiation）来加快求导。 

## 简单例子

**假设我们想对函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于列向量$\mathbf{x}$求导**

首先，我们创建变量`x`并为其分配一个初始值。

```python
import torch

x = torch.arange(4.0) # tensor([0., 1., 2., 3.])
```

将x设为需要求导

```python
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
```

计算$y=2\mathbf{x}^{\top}\mathbf{x}$并**通过调用反向传播函数来自动计算`y`关于`x`每个分量的梯度**

```python
y = 2 * torch.dot(x, x)

y.backward()
x.grad # tensor([ 0.,  4.,  8., 12.])
```

计算`x`的另一个函数

```python
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad # tensor([1., 1., 1., 1.])
```

## 非标量变量的backward

当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵。 对于高阶和高维的y和x，求导的结果可以是一个高阶张量。

当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。 这里，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。

```python
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad # tensor([0., 2., 4., 6.])
```

## 分离计算

例如，假设y是作为x的函数计算的，而z则是作为y和x的函数计算的。 我们想计算z关于x的梯度，但由于某种原因，**希望将y视为一个常数， 并且只考虑到x在y被计算后发挥的作用**。

这里可以分离`y`来返回一个新变量`u`，该变量与`y`具有相同的值， 但丢弃计算图中如何计算`y`的任何信息。 换句话说，梯度不会向后流经`u`到`x`。 因此，下面的反向传播函数计算`z=u*x`关于`x`的偏导数，同时将`u`作为常数处理， 而不是`z=x*x*x`关于`x`的偏导数。

```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

> 使用自动微分的一个好处是： 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。
