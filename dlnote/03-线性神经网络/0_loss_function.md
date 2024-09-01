# 损失函数

## L2 Loss

$$l(y,y') = \frac{1}{2}(y-y')^2$$

## L1 Loss

$$l(y,y') = |y-y'|$$

> 缺点：grad在0处突变，不连续

## Huber's Robust Loss

$$
L_{\delta}(a) =
\begin{cases}
\frac{1}{2} a^2 & \text{if } |a| \le \delta \\
\delta (|a| - \frac{1}{2} \delta) & \text{otherwise}
\end{cases}
$$

> 结合二者优点
