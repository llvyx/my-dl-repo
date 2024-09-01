# 数据操作

## 入门

张量 `tensor`

```python
import torch

# create a tensor
x = torch.arange(12)

# shape
x.shape

# number of elements
x.numel()

# reshape
x = x.reshape(3, 4)

# special elements
torch.zeros((2, 3, 4))

'''
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
'''

torch.ones((2, 3, 4))

# 每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
torch.randn(3, 4)

torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

> `reshape`和`view`方法的区别：
> 
> - `view`要求`tensor`必须连续
> - `tensor`连续时，二者都是软复制（共享数据内存）
> - `tensor`不连续时，`reshape`返回一个新的地址

## 运算符

### 基本运算

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算

# “按元素”方式可以应用更多的计算，包括像求幂这样的一元运算符。
torch.exp(x)
```

```text
(tensor([ 3.,  4.,  6., 10.]),
 tensor([-1.,  0.,  2.,  6.]),
 tensor([ 2.,  4.,  8., 16.]),
 tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 tensor([ 1.,  4., 16., 64.]))
```

### 连结（concatenate）

`cat` 方法

```python
X = torch.arange(12, dtype=torch.float32).reshape((1,3,4))
Y = torch.tensor([[[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=2)
```

### 通过逻辑运算符构建二元张量

```python
X == Y
```

out:

```text
tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
```

### 求和

```python
X.sum()
```

## 广播机制

在某些情况下，即使形状不同，我们仍然可以通过调用 广播机制（broadcasting mechanism）来执行按元素操作。这种机制的工作方式如下：

1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；

2. 对生成的数组执行按元素操作。

在大多数情况下，我们将沿着数组中长度为1的轴进行广播，如下例子：

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

out:

```text
(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]))
```

in:

```python
a + b
```

out:

```text

tensor([[0, 1],
        [1, 2],
        [2, 3]])
```

## 索引和切片

> 与`python`数组类似操作
>
> 为***多个元素赋值相同的值***，只需要**索引所有元素**，然后为它们赋值。

## 节省内存

```python
before = id(Y)
Y = Y + X
id(Y) == before
# false

before = id(Y)
Y += X
id(Y) == before
# true
```

solusion:

使用`Y[:] = <expression>`

```python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

## 转换为其他Python对象

### 转换为NumPy张量（ndarray）

```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
# (numpy.ndarray, torch.Tensor)
```

### 将大小为1的张量转换为Python标量

```python
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
# (tensor([3.5000]), 3.5, 3.5, 3)
```
