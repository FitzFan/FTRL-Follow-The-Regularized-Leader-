#!/usr/bin/env python
#coding:utf-8



"""
1、为什么会引入FTRL(Follow The Regularized Leader)？
- 传统的进行CTR预估的方法，比如LR，FM等，都属于“batching learning”，即离线的学习方法。统一的玩法都是：
-- 给一堆样本数据，也许很大；
-- 构建模型，LR or FM or others；
-- 选定optimizer，GD族 or ALS or MCMC or newton 
-- 求出最优解or局部最优解
-- save model and use model to predict；
-- 这就有一个很大的弊端：无法有效地处理大规模的在线数据流
-- 最后的结果就是：模型上线后，更新的周期会比较长（一般是一天，效率高的时候为一小时），这种模型上线后，一般是静态的（一段时间内不会改变），不会与线上的状况有任何互动，假设预测错了，只能在下一次更新的时候完成更正。


2、为什么要稀疏解：
- sgd 的训练出来的特征参数不具有稀疏性，从工程的角度占用内存过大。很多特征的权重其实很小，但因为是非0的也会占用内存空间。所以，需要更好的正则项来找到这些非0项


3、online学习中，给模型加L1为什么不work?
- 因为在 online 中，梯度的方向不是全局的方向。而是沿着样本的方向.... 那就会造成每次没有被充分训练的样本被错误的将系数归0了


4、学到一个构建正负例的方法“skip above”：
- 用户点击的item位置以上的展现才可能视作负例。
- 比如：假设用户点击了第i个位置，我们保留从第1条到第i+2条数据作为训练数据，其他的丢弃。这样能够最大程度的保证训练样本中的数据是被用户看到的。


5、Truncated Gradient Descent详解
- 资料：https://zhuanlan.zhihu.com/p/32903540、https://blog.csdn.net/google19890102/article/details/47422821
- paper：https://papers.nips.cc/paper/3585-sparse-online-learning-via-truncated-gradient.pdf  【看paper更直接，更棒】
- 通过参数g(gravity parameter) 和 theta 来控制模型输出解的稀疏性。
- 比简单截断法要进步一些，简单截断法的玩法：
-- 系数的值小于预设阈值，直接设为0
-- 和L1范数有得一拼；
-- 但会有一个隐患：某特征的系数比较小可能是因为该维度训练不足引起的，简单进行截断会造成这部分特征的丢失。
- Truncated Gradient Descent的玩法：
-- 只在时间窗口k下，进行truncate；
-- truncate的对象是每一维特征的参数值，要么和SVD一样，要么为0；
-- 根据基于梯度下降更新后的参数值，进行truncate，具体方式参见paper；
-- 看公式可以知道：
	a) Truncated Gradient Descent对参数的约束比较强：会对参数值施以一个gravity parameter的惩罚，并且gravity parameter随着迭代次数逐渐增加，提高约束性；
	b) 基于此，Truncated Gradient Descent 达到了稀疏求解的效果。
	c) 特别地，如果theta值无限大，那么在每一个可以shrink的iteration，都会进行稀疏约束。
	d) 从公式上看，当gravity parameter 或learning_rate逼近于0时，TSVD均逼近退化为L1-SubGradient。
	e) 从公式上看，要完全使TG退化为L1-SubGradient，必须还加上K=1这个条件。


6、FOBOS
- 资料：https://stanford.edu/~jduchi/projects/DuchiSi09b.pdf【看这篇paper就足够了】
- 主要有两项：第一项是标准梯度下降；第二项是微调，处理正则化，产生稀疏性
- 关于g^f_t 和 g^r_{t+1}的含义在paper中有解释：
-- g^f_t 就是t时刻的梯度；
-- g^r_{t+1}是使那一坨等式等于0的r(w)的偏导值；
-- η_{t+1/2}的含义：当是online背景时η_{t+1/2} = η_t，当是离线（batch）背景时，η_{t+1/2} = η_{t+1}
- 根据资料，个人只推导出：
-- 当L1-FOBOS的lambda = TG中的gravity，且theta = +∞，且K=1时，二者coincide；


7、**好开心：学习FOBOS，顺便理清了为什么L1可以带来稀疏，而L2不会？
- 资料：chrome-extension://cbnaodkpfinfiipjblikofhlhlcickei/src/pdfviewer/web/viewer.html?file=https://stanford.edu/~jduchi/projects/DuchiSi09b.pdf
- 根据上述资料的公式（6）和公式（7）就能推导出L1和L2的性质以及lambda 的值是如何起作用的。


8、小结：
- TG和FOBOS都是基于SGD的玩法，即在使用gradient descent的思想下，各显神通，保证求出稀疏解。
- 而下面的RDA是进行在线求解，并且有效提升了特征权重的稀疏性。


9、RDA（Regularized Dual Averaging Algorithm） and RDA with L1


10、FTRL


11、FTRL-Proximal




"""


