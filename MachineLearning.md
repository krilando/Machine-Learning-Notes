# Machine Learning

## 1 机器学习分类
### 1.1 Supervised Learning
拟合input和output，从而根据input预测output
1. Regression 回归: 从无限多可能的输出中预测输出
2. Classification: 预测分类，有限输出
### 1.2 Unsupervised Learning
数据没有明确的label，并不需要进行预测
1. Clustering 聚类：比如“相关推荐”，通过将没有label的数据，即在没有人指导的情况下根据特征进行分组（grouping them into clusters）
2. Anomaly Detection 异常检测：检测异常事件
3. Dimensionality Reduction 降维：压缩极大的数据集

## 2 Linear Regression
### 2.1 Training set: 
Notation:
1. x = "input" variable = feature
2. y = "output" variable = target
3. m = number of training examples
4. (x,y) = single training examples
5. $(x^{(i)},y^{(i)})$ = i^th^ training example
### 2.2 Univariate Linear Regression
Linear Regression with one variable:
$$
f_{w,b}(x)=wx+b=\widehat{y}
$$
- parameters/coefficient/weight: w.b
- slope 斜率
- intercept 截距
### 2.3 Cost Function
Squared error cost function:
$$
J(w,b)=\frac{1}{2m}\sum_{i=1}^m (\widehat{y}^{(i)} - y^{(i)})^2
$$
- 可视化代价函数：利用contour plots等高线图，w，b为x，y轴，J(w,b)为z轴
### 2.4 Linear Regression中的Gradient Descent
$\begin{gathered}
f_{w,b}(x^{(i)})=wx^{(i)}+b=\widehat{y}^{(i)}\\
J(w,b)=\frac{1}{2m}\sum_{i=1}^m (\widehat{y}^{(i)} - y^{(i)})^2\\
w=w-\alpha \frac{\partial J(w,b)}{\partial {w}}\\
b=b-\alpha \frac{\partial J(w,b)}{\partial {b}}\\
\end{gathered}$

代入：复合函数求偏导

$$
\begin{aligned}
w &= w-\alpha \frac{\partial J(w,b)}{\partial {w}}\\
&= w-\alpha \frac{1}{2m}\sum_{i=1}^m (\widehat{y}^{(i)} - y^{(i)})*2\frac{\partial \widehat{y}^{(i)}}{\partial w} \\
&= w-\alpha \frac{1}{m}\sum_{i=1}^m [(\widehat{y}^{(i)}-y^{(i)})*x^{(i)}]\\
\end{aligned}
$$
$$
\begin{aligned}
b&=b-\alpha \frac{\partial J(w,b)}{\partial {b}}\\
&=b-\alpha \frac{1}{2m}\sum_{i=1}^m (\widehat{y}^{(i)} - y^{(i)})*2\frac{\partial \widehat{y}^{(i)}}{\partial b}\\
&= b-\alpha \frac{1}{m}\sum_{i=1}^m (\widehat{y}^{(i)}-y^{(i)})
\end{aligned}
$$


## 3 Gradient Descent 梯度下降
作用：求local minimum
Simultaneous update of w and b:
$$
\begin{aligned}
& tmp\_w=w-\alpha \frac{\partial J(w,b)}{\partial {w}}\\
& tmp\_b=b-\alpha \frac{\partial J(w,b)}{\partial {b}}\\
& w=tmp\_w\\
& b=tmp\_b\\
\end{aligned}
$$

### 3.1 Learning Rate:$\alpha$
取值：(0,1)
如果太小，gradient descent算法执行效率低
如果太大，找不大minimum

让w无限趋近local minimum的自变量取值：
![](gradientDescent.png)

### 3.2 分类
1. Batch：每轮计算都用到所有样本，适用于小样本

## 4 Multiple Features