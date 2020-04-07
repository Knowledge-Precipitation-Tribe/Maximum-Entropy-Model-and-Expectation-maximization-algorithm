# Maximum-Entropy-Model-and-Expectation-maximization-algorithm
Maximum Entropy Model and Expectation-maximization algorithm，最大熵模型与EM算法。

## 阅读指南

1. 在线观看请使用Chrome浏览器，并安装插件：[MathJax Plugin for Github(需科学上网)](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)， 插件[Github地址](https://github.com/orsharir/github-mathjax)
2. 或下载内容到本地，使用markdown相关软件打开，如：[Typora](https://typora.io/)
3. **若数学公式显示出现问题大家也可通过jupyter notebook链接查看：[Maximum-Entropy-Model-and-Expectation-maximization-algorithm](https://nbviewer.jupyter.org/github/Knowledge-Precipitation-Tribe/Maximum-Entropy-Model-and-Expectation-maximization-algorithm/blob/master/jupyter notebook/Maximum-Entropy-Model-and-Expectation-maximization-algorithm.ipynb)**

## content



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



## [最大熵模型](#content)



## [参考文献](#content)



