# 数据预处理

> 本节将简要介绍使用`pandas`预处理原始数据，并将原始数据转换为张量格式的步骤。

## 读取数据集

### 创建数据集

> 创建一个人工数据集，并存储在CSV（逗号分隔值）文件`../data/house_tiny.csv`中。

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

### 加载数据集

```python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 处理缺失值

> 为了处理缺失的数据，典型的方法包括`插值法`和`删除法`， 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。 这里，我们将考虑插值法。
>
> 通过位置索引`iloc`(index location)，我们将`data`分成`inputs`和`outputs`， 其中前者为`data`的前两列，而后者为`data`的最后一列。 对于inputs中缺少的`数值`，我们用同一列的均值替换“`NaN`”项。

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only = True))
print(inputs)
```

```text
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
```

> **对于inputs中的`类别值`或`离散值`，我们将“NaN”视为一个类别。** 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

```text
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```

## 转换为张量格式

> 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。

```python
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```
