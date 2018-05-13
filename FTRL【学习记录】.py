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
- **online学习的特点：
-- 每来一个训练样本，就用该样本产生的loss和梯度对模型迭代一次，一个一个数据地进行训练，因此可以处理大数据量训练和在线训练。


2、为什么要稀疏解：
- sgd 的训练出来的特征参数不具有稀疏性，从工程的角度占用内存过大。很多特征的权重其实很小，但因为是非0的也会占用内存空间。所以，需要更好的正则项来找到这些非0项


3、online学习中，给模型加L1为什么不work?
- 因为在 online 中，梯度的方向不是全局的方向。而是沿着样本的方向.... 那就会造成每次没有被充分训练的样本被错误的将系数归0了
- L1在离线学习中，还是很work的。


4、学到一个构建正负例的方法“skip above”：
- 用户点击的item位置以上的展现才可能视作负例。
- 比如：假设用户点击了第i个位置，我们保留从第1条到第i+2条数据作为训练数据，其他的丢弃。这样能够最大程度的保证训练样本中的数据是被用户看到的。


5、Truncated Gradient Descent详解
1) 资料：https://zhuanlan.zhihu.com/p/32903540、https://blog.csdn.net/google19890102/article/details/47422821
2) paper：https://papers.nips.cc/paper/3585-sparse-online-learning-via-truncated-gradient.pdf  【看paper更直接，更棒】
3) 通过参数g(gravity parameter) 和 theta 来控制模型输出解的稀疏性。
4) 简单截断法(Simple Coefficient Rounding)
-- 系数的值小于预设阈值，直接设为0
-- 但会有一个隐患：某特征的系数比较小可能是因为该维度训练不足引起的，简单进行截断会造成这部分特征的丢失。
5) L1-Regularized Subgradient:
-- 在进行梯度下降时，施加L1范数；
-- 直观理解就是：对每一次的梯度更新，将下降的步伐缩减lambda;
-- 达到的效果是：容易衰减为0，但考虑到精度问题，如果不做任何特殊处理，是会衰减到0的附近。【强烈怀疑现在机器学习关于L1的实现是加入了Simple Coefficient Rounding的】
6) Truncated Gradient Descent的玩法：
-- 只在时间窗口k下，进行truncate；
-- 根据基于梯度下降更新后的参数值，进行truncate，具体方式参见paper；
-- 看公式可以知道：
	a) Truncated Gradient Descent对参数的约束比较强：会对参数值施以一个gravity parameter的惩罚，基于此，Truncated Gradient Descent 达到了稀疏求解的效果。
	b) Truncated Gradient Descent和L1-Regularized Subgradient有本质的不同：Truncated Gradient Descent不允许参数在迭代过程中改变正负号。
	c) 所以，Truncated Gradient Descent和L1-Regularized Subgradient永远不可能一样。
	d) 但是，Truncated Gradient Descent可以和L1-Regularized Subgradient+Simple Coefficient Rounding一样。
	e) 特别地，如果theta值无限大，那么在每一个可以shrink的iteration，都会进行稀疏约束。
	f) 从公式上看，当gravity parameter 或learning_rate逼近于0时，TSVD均逼近退化为L1-SubGradient+Simple Coefficient Rounding。
	g) 从公式上看，要完全使TG退化为L1-SubGradient+Simple Coefficient Rounding，必须还加上K=1这个条件。

7) **Truncated Gradient Descent本质总结**
-- 每K轮迭代，对特征的参数值施以L1惩罚，以此达到稀疏的效果
-- 数学表达见paper的公式(6)，公式(6)表达出TG不允许参数在迭代过程中改变正负号。


6、FOBOS（前向后向切分）
1) 资料：https://stanford.edu/~jduchi/projects/DuchiSi09b.pdf【看这篇paper就足够了】
2) 主要有两项：第一项是标准梯度下降；第二项是微调，处理正则化，产生稀疏性
3) 关于g^f_t 和 g^r_{t+1}的含义在paper中有解释：
-- g^f_t 就是t时刻的梯度；
-- g^r_{t+1}是使那一坨等式等于0的r(w)的偏导值；
-- η_{t+1/2}的含义：当是online背景时η_{t+1/2} = η_t，当是离线（batch）背景时，η_{t+1/2} = η_{t+1}
4) 根据资料，个人只推导出：
-- 当L1-FOBOS的lambda = TG中的gravity，且theta = +∞，且K=1时，二者coincide；
-- 为什么有这种coincice: **将paper中的公式(6)的绝对值符号拆成分段函数，就和TG的分段函数一样了。
5) 名字的由来：
-- 前一个步骤是一个标准的梯度下降，后一个步骤可以理解为对梯度下降结果进行微调。
-- 对W的微调也分为两部分：前一部分保证微调发生在梯度下降结果的附近，后一部分则用于处理正则化产生稀疏性。
6) 为什么零向量一定属于F(W)的次梯度？
-- 其实整个FOBOS的形式就是用MSE+L1作为loss function来建模；
-- 按照常规玩法，令其导数为零即可求出最优解；
-- 所以才有paper公式(6)的关于此时使用L1作为正则项的最优解形式；
7) **当FOBOS with Regularizer的本质：
-- 首先用SGD算出本轮梯度下降的结果，当作因变量。然后用MSE+Regularizer作为Loss Function进行建模求解。
-- 由于零向量一定是这个Loss Function 导数的驻点。
-- 
-- 如果这次训练产生梯度的变化不足以令权重值发生足够大的变化时，就认为在这次训练中该维度不够重要，应该强制其权重是 0 ；
-- 衡量变化大小的标尺是正则项；
8) L1-FOBOS 本质上是 TG在某种特殊情况下的形式
9) 【*******】如何用KKT条件求出FOBOS-L1的最优解形式？？？？？？如果弄懂这个会弄懂一堆问题。
-- 首先最优解一定是0；
-- 那么怎么根据KKT求出拉格朗日乘子非零的解？？？



7、**好开心：学习FOBOS，顺便理清了为什么L1可以带来稀疏，而L2不会？
- 资料：https://stanford.edu/~jduchi/projects/DuchiSi09b.pdf
- 根据上述资料的公式（6）和公式（7）就能推导出L1和L2的性质以及lambda 的值是如何起作用的。
- 主要模型优化的方向都是min()，所以L1正则的形式会是公式(6)
- 和青爷沟通了后发现：在Model的Loss function中加入L1是大概率会产生稀疏解，但并没有说一定。
- 因为有精度的问题，一般情况下是比较难产生稀疏解的。除非有使用“Simple Coefficient Rounding”。
- L1是容易产生稀疏解，是相对L2而言。
- L1正则项是为了使得那些原先处于零（即|w|≈0）附近的参数w往零移动，使得部分参数为零。
- L2在理论上是不可能产出稀疏解，因为lambda不可能为无穷大。L2是产出平滑的约束解；


8、纠结了2天的加入了L1正则的求解方法：
1) L1正则求解本质上是进行分段讨论，具体求解过程见自己做的笔记；
2) L1正则最后求解的结果形式，是一个以lambda为阈值的软阈值函数；
3) 软阈值和硬阈值的区别：
-- 硬阈值只分两段：且其中一段一定是0
-- 软阈值可以分为3段：其中一段也是0


9、小结：
1) TG和FOBOS都是基于SGD的玩法，即在使用gradient descent的思想下，各显神通，保证求出稀疏解。
2) 而下面的RDA是进行在线求解，并且有效提升了特征权重的稀疏性。

---------------------------------------------Is Gradient Descent Or Not-------------------------------------------------------

10、RDA（Regularized Dual Averaging Algorithm） and RDA with L1
1) L1-RDA的截断阈值是常数λ,并不随t而变化，因此可以认为是L1-RDA比L1-FOBOS更武断，这种性质使得它更容易产生稀疏性。
2) 此外，RDA中判定对象是梯度的累加平均值，不同于之前的针对单次梯度的计算结果进行判定，避免了某些维度由于训练不足导致的截断问题。
3) 通过调节λ很容易在精度和稀疏性上进行权衡。
4) RDA with L1:
-- 可以看到当梯度积累的平均值小于阈值lambda的时候就归0了，从而产生特征权重的稀疏性。
-- RDA with L1，其实是同时使用了L1和L2正则来对梯度进行约束。
-- RDA with L1并没有像FOBOS-L1那样要求调整后的梯度必须离标准SGD的梯度太远；
-- RDA with L1 要求的是调整后的梯度不能离 0 太远，所以RDA with L1更能产生稀疏解；
5) **RDA会加入额外项，目的有 2 个：
- 和L1一起作为正则项，来约束复杂度；
- 更重要的是：让目标函数具有最小值！！【如果不加入这几个L2范数作为额外项，RDA是一条直线，木有最小值。求导后自变量都木有了！】
6) Loss = 累积梯度 + L1 + 额外的L2

7) 总结：
-- 从截断方式来看，在 RDA 的算法中，只要梯度的累加平均值小于参数lambda, 就直接进行截断，说明 RDA 更容易产生稀疏性；
-- 同时，RDA中截断的条件是考虑梯度的累加平均值，可以避免因为某些维度训练不足而导致截断的问题，这一点与 TG，FOBOS 不一样;
-- 还有一点不一样是：TG和FOBOS都要求离调整前的梯度不要太远，而RDA是要求不要离 0 太远。
-- ***FOBOS-L1的“截断阈值”岁时间t的变化而减小，所以稀疏性没有RDA-L1那么好！
-- ***RDA-L1的“截断阈值”为λ，不随t变化，截断判定更加aggressive，稀疏性更好！
-- 判定对象是之前所有的梯度累加均值，而不是单次梯度，避免了由于某些维度由于训练不足导致截断的问题，更加合理！

11、FTRL（Follow The R egularized Leader Proximal）
1) 预备知识：
（1）FOBOS-L1
	- FOBOS-L1具有较高的精度，因为它限定调整后的梯度不能离SGD的结果太远，即不能离使loss function值最低的距离太远。
	- FOBOS-L1但由于阶段阈值随迭代次数的衰减，所以稀疏性不够。
	- FOBOS-L1的判定对象是当前梯度，可能会由于训练不足导致截断问题，某种程度上可能会失去一些重要特征，降低模型的精度；
（2）RDA-L1
	- RDA-L1具有较好的稀疏性，因为它限定调整后的梯度不能离 0 太远，但牺牲了精度；
	- RDA-L1判断的对象是过往累积梯度的均值，避免由于训练不足导致截断问题；
（3）paper：
	- https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41159.pdf
2) FTRL综合了FOBOS-L1和 RDA-L1的优势：
	- 使用L1正则项产出稀疏解，使用RDA中的累积梯度，帮助产出有效稀疏解。
3) **从loss function的形式来看：FTRL就是将RDA-L1的“梯度累加”思想应用在FOBOS-L1上。，这样达到的效果是：
	- 累积加和限定了新的迭代结果W不要离“已迭代过的解”太远；
	- 因为调整后的解不会离迭代过的解太远，所以保证了每次找到让之前所有损失函数之和最小的参数；
	- 保留的RDA-L1中关于累积梯度的项，可以看作是当前特征对损失函数的贡献的一个估计【累积梯度越大，贡献越大。】
	- 由于使用了累积梯度，即使某一次迭代使某个重要特征约束为0，但如果后面这个特征慢慢变得稠密，它的参数又会变为非0；
	- 保留的RDA-L1中关于累积梯度的项，与v相加，总会比原来的v大，加起来的绝对值更容易大于L1的阈值，保护了重要的特征；
4) **FTRL的巧妙之处在于**：
	- 在MSE的前面乘以了一个和learning_rate有着神奇关系的参数σ_s。
	- 因为这个参数，保证了FTRL在不使用L1时和SGD保持了一致性。
5) FTRL使用的自适应learning_rate，其思想和 Adagrad Optimizer 类似的自适应思想：
	- 如果特征稀疏，learning_rate就大一点；
	- 如果特征稠密，learning_rate就小一点；
6) **FTRL中为什么要同时兼顾FOBOS-L1和RDA-L1？？
	- 因为不是为了产出稀疏而进行变化，真正的目的是产出有效的稀疏解。即稀疏又保留有效特征！！！
	- 稀疏靠RDA-L1，保留有效特征靠FOBOS-L1和RDA-L1的累积梯度思想。
7) Loss = RDA_累积梯度 + FOBOS-L1 * 一个神奇的参数
8) **本质上，FTRL只是一种适用于online-learning的optimizer；
9) ***再总结一遍FTRL是如何结合FOBOS-L1和RDA-L1的，感觉还是有些乱。
	- FOBOS-L1：使用MSE+L1对w_{t+1/2}进行建模，目标是使调整后的梯度在离SGD结果附近的基础上，产出稀疏解；
	- RDA-L1：使用累积平均梯度 + L1 + L2进行建模，这里使用L2有两方面的理解：
	（1）能产出极小值点；
	（2）调整后的梯度不能与零点太远；
10) FTRL的工程实现paper【解释了PPT中提到的trick的缘由，必须强撸，面试可用】：
	- 资料1：https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41159.pdf
	- 资料2：https://sensirly.github.io/from-logistic-regression-to-ftrl/
	- 具体的tricks：【有些trick太工程了~】
	（1）使用正负样本的数目来计算梯度的和（所有的model具有同样的N和P）【为什么可以这么做？PPT中有数学推导】
	（2）probabilistic feature inclusion
		- 背景：在CTR中，特征仅仅出现了一次(In fact, in some of our models, half the unique features occur only once in the entire training set of billions of examples),  这样特征几乎没有什么用，但是存储起来非常浪费空间。L1正则化虽然解决了一些问题，但是这样降低了一些精度。谷歌使用基于“probabilistic feature inclusion”的方法进行特征选择：一个特征第一次出现时，会以一定的概率被保存使用。具体包括：
		- Poisson Inclusion：当一个新特征出现时，以固定概率P接受并更新
		- Bloom Filter Inclusion：用一系列的Bloom flters（布隆过滤器）来检测特征的前n次出现，一旦检测到出现了n次，就加入模型并用在后面的训练中。
	（3）Encoding Values with Fewer Bits【系数编码】【不具有适用性，可以参考下】
		- 对于谷歌的模型而言，几乎所有的权重值都在(-2, +2)之间。谷歌使用了q2.13 coding的方式而不是采用float。
		- q2.13编码，是用两个比特表示整数位数字，13比特表示小数位数字，1个比特表示正负号，一共需要16bit。这种方式还需要加入一点随机，否则可能导致误差的累加。
	（4）训练数据子采样 
		- 负样本按r的概率采样在训练时乘一个1/r的权重来弥补负样本的缺失。
		- 正样本按常规对待；
11) FTRL中关于learning_rate的设置：
	- 设置Per-Coordinate变量，它根据每个特征在样本中出现的次数来推算它的学习率。
	- 如果出现的次数多，那么模型在该特征上学到的参数就已经比较可信了，所以学习率可以不用那么高；
	- 而对于出现次数少的特征，认为在这个特征上的参数还没有学完全，所以要保持较高的学习率来使之尽快适应新的数据。

"""


