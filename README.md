# Maximum-Entropy-Model-and-Expectation-maximization-algorithm
Maximum Entropy Model and Expectation-maximization algorithm，最大熵模型与EM算法。

## 阅读指南

1. 在线观看请使用Chrome浏览器，并安装插件：[MathJax Plugin for Github(需科学上网)](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)， 插件[Github地址](https://github.com/orsharir/github-mathjax)
2. 或下载内容到本地，使用markdown相关软件打开，如：[Typora](https://typora.io/)
3. **若数学公式显示出现问题大家也可通过jupyter notebook链接查看：[Maximum-Entropy-Model-and-Expectation-maximization-algorithm](https://nbviewer.jupyter.org/github/Knowledge-Precipitation-Tribe/Maximum-Entropy-Model-and-Expectation-maximization-algorithm/blob/master/jupyter%20notebook/Maximum-Entropy-Model-and-Expectation-maximization-algorithm.ipynb)**

## Content

- <a href = "#熵等相关概念">熵等相关概念</a>
  - <a href = "#信息量">信息量</a>
  - <a href = "#熵">熵</a>
    - <a href = "#联合熵">联合熵</a>
    - <a href = "#条件熵">条件熵</a>
    - <a href = "#互信息">互信息</a>
  - <a href = "#相对熵">相对熵</a>
  - <a href = "#交叉熵">交叉熵</a>
- <a href = "#最大熵模型">最大熵模型</a>
- <a href = "#极大似然估计">极大似然估计</a>
- <a href = "#Jensen不等式">Jensen不等式</a>
- <a href = "#EM算法">EM算法</a>



# [熵等相关概念](#content)

## [信息量](#content)

信息的度量，一个是发生的概率越高，那么它的信息量就越低。

## [熵](#content)

熵是信息量的期望（用一个大球来表示，）

反映的是不确定性，变量发生的概率越低，不确定性越高，熵越高。

### [联合熵](#content)

用韦恩图表示，就是两个圆相交的总体面积

联合熵公式
$$
H(x, y)=-\sum_{i=1}^{n} \sum_{j=1}^{n} f\left(x_{i}, y_{j}\right) \log _{2} p\left(x_{i}, y_{j}\right)
$$
这就是

### [条件熵](#content)

用韦恩图表示，就是大圆减去交集部分

### [互信息](#content)

用韦恩图表示就是两个圆的交集，

用于特征选择或特征间的关联性，然后做降维处理

## [相对熵](#content)

相对熵有名KL散度，刻画的事两个分布之间的差异（KL散度，Wase， MMD）。

## [交叉熵](#content)



# [最大熵模型](#content)



# [极大似然估计](#content)



# [Jensen不等式](#content)



# [EM算法](#content)

## [什么是EM算法](#content)

EM算法被称为数据挖掘的十大算法之一，在机器学习和数据挖掘领域占有很重要的地位。接下来我就用尽量通俗的语言来梳理一下EM算法。EM算法也称为期望最大化(Expectation-Maximum)算法，在很多地方都能看到他的身影，例如HMM以及LDA等。

> 维基百科定义：
>
> **最大期望算法**（**Expectation-maximization algorithm**，又译**期望最大化算法**）在统计中被用于寻找，依赖于不可观察的隐性变量的概率模型中，参数的最大似然估计。
>
> 在[统计](https://zh.wikipedia.org/wiki/统计)[计算](https://zh.wikipedia.org/wiki/计算)中，**最大期望（EM）算法**是在[概率模型](https://zh.wikipedia.org/wiki/概率模型)中寻找[参数](https://zh.wikipedia.org/wiki/参数)[最大似然估计](https://zh.wikipedia.org/wiki/最大似然估计)或者[最大后验估计](https://zh.wikipedia.org/wiki/最大后验概率)的[算法](https://zh.wikipedia.org/wiki/算法)，其中概率模型依赖于无法观测的[隐变量](https://zh.wikipedia.org/wiki/隐变量)。最大期望算法经常用在[机器学习](https://zh.wikipedia.org/wiki/机器学习)和[计算机视觉](https://zh.wikipedia.org/wiki/计算机视觉)的[数据聚类](https://zh.wikipedia.org/wiki/数据聚类)（Data Clustering）领域。最大期望算法经过两个步骤交替进行计算，第一步是计算期望（E），利用对隐藏变量的现有估计值，计算其最大似然估计值；第二步是最大化（M），最大化在E步上求得的[最大似然值](https://zh.wikipedia.org/wiki/最大似然估计)来计算参数的值。M步上找到的参数估计值被用于下一个E步计算中，这个过程不断交替进行。

看完是不是有种黑人问号脸的感觉。

![heirenwenhao](img/heirenwenhao.jpg)

但是不要慌，我们接下来就来梳理一下这个EM算法。

## [EM算法通俗解释](#content)

在一些资料中首先会介绍一下什么是似然函数，什么是极大似然估计等等，我们这里先试着抛开这些东西不谈，直接来理解一下EM算法。

**EM算法用最简单的话来说就是：给你一堆数据，然后告诉你这些数据是满足一个分布的，现在你就来推导一下参数是什么样的分布可以拟合这堆数据。**

如果你了解过参数估计，那你可能会有一点头绪，就算没有头绪也没关系，现在我们来看一个例子。

### [例子](#content)

例子可能并不那么严谨，欢迎批评指正。

现在我们有4位志愿者，我们通过测量获得了他们的身高数据。但是这些志愿者当中有男性有女性，具体比例我们没有记录，所以不知道男女各有多少。假设男性和女性的身高分别服从$\mathrm{N}\left(\mu_{男}, \sigma_{男}\right)$和$\mathrm{N}\left(\mu_{女}, \sigma_{女}\right)$的高斯分布，现在我们来估计一下这四个参数$\mu_{男}, \sigma_{男},\mu_{女}, \sigma_{女}$可能的值是多少。

解：

反正现在是让我们估计这几个参数，我们不妨瞎猜一下，就假设$\mu_{男}=170, \sigma_{男}=6,\mu_{女}=160, \sigma_{女}=5$，而且男女各一半我们用$\pi_{男}=\pi_{女}=0.5$表示。

而且已知其是满足高斯分布的，那么我们可以得到高斯分布的概率密度函数：
$$
f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}}
$$
现在我们拿到第一个志愿者的身高数据：$x_1=1.88$，我们现在来判断一下第一个志愿者属于男性的概率有多大。我们现在将1.88代入到概率密度函数当中
$$
\begin{aligned}
f(188 | \mu_{男}, \sigma_{男}) &=\frac{1}{6 \sqrt{2 \pi}} e^{-\frac{(188-170)^{2}}{2 \times 6^{2}}} \\
&= 7.368 \times 10^{-4} \\
f(188 | \mu_{女}, \sigma_{女}) &=\frac{1}{5 \times \sqrt{2 \pi}} e^{-\frac{(188-160)^{2}}{2 \times 5^{2}}} \\
&= 1.236 \times 10^{-8}
\end{aligned}
$$
而且我们假设$\pi_{男}=\pi_{女}=0.5$，那么当前第一个志愿者为男性的概率
$$
\frac{7.368 \times 10^{-4} \times 0.5}{7.368 \times 10^{-4} \times 0.5+1.236 \times 10^{-8} \times 0.5}=0.999
$$
为女性的概率
$$
\frac{1.236 \times 10^{-8} \times 0.5}{7.368 \times 10^{-4} \times 0.5+1.236 \times 10^{-8} \times 0.5}=0.001
$$
看来我们猜的还可以，1.88属于男性的概率确实很高。Anyway，我们继续来算一下剩下的三个志愿者的数据，与第一个志愿的算法类似，这里就不再赘述了。

剩下的三个志愿者的身高数据分别为：$x_2=158, x_3=165, x_4=170$。那我们就可以得到这样一组数据。



之后



## [参考文献](#content)

[1] 刘建平：[最大熵模型原理小结](https://www.cnblogs.com/pinard/p/6093948.html)

[2] 忆臻: [一文搞懂极大似然估计](https://zhuanlan.zhihu.com/p/26614750)

[3] 知行流浪：[极大似然估计详解](https://blog.csdn.net/zengxiantao1994/article/details/72787849)

[4] 刘建平：[EM算法原理总结](https://www.cnblogs.com/pinard/p/6912636.html)

[5] v_JULY_v：[如何通俗理解EM算法](https://blog.csdn.net/v_JULY_v/article/details/81708386)

