# 深度学习：神经网络

**神经网络（Neural Network）**的构筑理念是受到生物神经网络功能的运作启发而产生的。人工神经网络通常是通过一个基于数学统计学类型的学习方法得以优化，所以人工神经网络也是数学统计学方法的一种实际应用。

和其他机器学习方法一样，神经网络已经被用于解决各种各样的问题，例如机器视觉和语音识别。这些问题都是很难被传统基于规则的编程所解决的。

<a id="more"></a>
### 什么是神经网络？

机器学习领域所说的神经网络指的是一种模仿生物神经网络的结构和功能而建立的数学或计算模型，用于对函数进行估计或近似。

例如，给定一些关于市面上房子的面积及价格的数据，要你根据这些数据建立一个房价预测模型，即输入一个房子的面积，希望通过这个模型输出一个房价的预测值。显然，这是一个线性回归问题，因为一般情况下房价和房子的面积都成正相关。这时，我们可以将已知数据的关系表现在平面坐标系中：
![房价线性模型](https://ws1.sinaimg.cn/large/82e16446ly1fjtjyyfqxrj20f10b00sy.jpg)
对数据进行线性拟合，且房价永远不会是负数，得到图中的**ReLU函数（Rectified Linear Unit，修正线性单元）**。
![房价神经网络模型](https://ws1.sinaimg.cn/large/82e16446ly1fjtj8ibkakj20b105jdft.jpg)
在这个简单的例子中，房子的面积作为输入，房价作为输出，而ReLU函数便充当一个神经元的作用，来产生输出。

然而房价除了受房子的面积影响之外，还会受卧室的数量、房子的位置以及地区的财富水平等因素的影响，这时就需要构建一个更为复杂的神经网络模型。
![神经网络模型](https://ws1.sinaimg.cn/large/82e16446ly1fjtjl5miuuj20le0a2wfj.jpg)
这就构成了一个神经网络模型基本结构，神经网络会自动生成**隐藏层（Hidden Units）**来处理输入，生成输出。这个问题中，只要拥有足够的训练数据，就能生成一个较好的神经网络模型，得到较为精确的结果。

简单而言，深度学习便是更为复杂的神经网络。

在Logistic回归问题中，我们通过建立一个简单的神经网络模型，输入训练样本(x,y)，希望得出一个预测值ŷ y^，使得ŷ y^尽可能等于y。训练的流程如下：
![Logistic 回归流程](https://ws1.sinaimg.cn/large/82e16446ly1fjtl1m6yfhj20cp0a90sx.jpg)
在这个模型中，我们先建立损失函数，进而不断采用梯度下降法找到参数w和b的最优解。采用这种算法编写的猫识别器最终的准确率只有70%，想要进一步提高识别的精准度，就需要建立起一个多层的神经网络来训练样本。

### 符号约定

如图所示的神经网络中，前面为输入层，中间为隐藏层 ，最后为输出层。中间层被称为隐藏层的原因是因为在训练过程中，将看到输入的样本有哪些，输出的结果是什么，中间层中的神经节点产生的真实值无法被观察到。所以中间层被称为隐藏层，只是因为你不会在训练集中看到它。

![两层神经网络](https://ws1.sinaimg.cn/large/82e16446ly1fjthu97petj20ix0dimyp.jpg)

此前，我们使用特征向量X来表示输入，在此前我们用符号a[0]a[0]来替代，上标“[ ]”括号中的数字表示神经网络中的第几层，而a代表着激活（Activation），指的是不同层次的神经网络传递给后续层次的值。将输入集传递给隐藏层后，隐藏层随之产生激活表示为a[1]a[1]，而隐藏层的第一节点生成的激活表示为a[1]1a1[1]，第二个节点产生的激活为a[1]2a2[1]，以此类推，则：

 a[1]=⎡⎣⎢⎢⎢⎢⎢a[1]1a[1]2a[1]3a[1]4⎤⎦⎥⎥⎥⎥⎥a[1]=[a1[1]a2[1]a3[1]a4[1]]

最后，输出层输出的值表示为a[2]a[2]，则ŷ y^ = a[2]a[2]。

神经网络中的符号约定中，方括号上标明确指出了激活a来源于哪一层，而且，图中的这个神经网络也被称为两层神经网络，原因是当我么计算神经网络的层数时，通常不计算输入层。所以这个神经网络中，隐藏层是第一次层，输出层是第二层，而输入层为第零层。

图中的隐藏层中，将存在参数w和b，它们将分别表示为w[1]w[1]和b[1]b[1]，w[1]w[1]将会是个4*3矩阵，b[1]b[1]将会是个4*1矩阵。输出层中，也会存在参数w[2]w[2]和b[2]b[2]，w[2]w[2]是个1*4矩阵，b[2]b[2]是个1*1矩阵。

### 神经网络的表示

![神经网络的表示](https://ws1.sinaimg.cn/large/82e16446ly1fjufdpan63j20ex09imxx.jpg)

如图所示，将样本输入隐藏层中的第一个节点后，可得；

	 z[1]1=w[1]T1X+b[1]1,a[1]1=σ(z[1]1)z1[1]=w1[1]TX+b1[1],a1[1]=σ(z1[1])

以此类推：

	 z[1]2=w[1]T2X+b[1]2,a[1]2=σ(z[1]2)z2[1]=w2[1]TX+b2[1],a2[1]=σ(z2[1])
	
	 z[1]3=w[1]T3X+b[1]3,a[1]3=σ(z[1]3)z3[1]=w3[1]TX+b3[1],a3[1]=σ(z3[1])
	
	 z[1]4=w[1]T4X+b[1]4,a[1]4=σ(z[1]4)z4[1]=w4[1]TX+b4[1],a4[1]=σ(z4[1])

将它们都表示成矩阵形式：

	 z[1]=⎡⎣⎢⎢⎢⎢⎢w[1]1w[1]2w[1]3w[1]4w[1]1w[1]2w[1]3w[1]4w[1]1w[1]2w[1]3w[1]4⎤⎦⎥⎥⎥⎥⎥⎡⎣⎢⎢x1x2x3⎤⎦⎥⎥+⎡⎣⎢⎢⎢⎢⎢b[1]1b[1]2b[1]3b[1]4⎤⎦⎥⎥⎥⎥⎥=⎡⎣⎢⎢⎢⎢⎢z[1]1z[1]2z[1]3z[1]4⎤⎦⎥⎥⎥⎥⎥z[1]=[w1[1]w1[1]w1[1]w2[1]w2[1]w2[1]w3[1]w3[1]w3[1]w4[1]w4[1]w4[1]][x1x2x3]+[b1[1]b2[1]b3[1]b4[1]]=[z1[1]z2[1]z3[1]z4[1]]

即：

	 z[1]=w[1]X+b[1]z[1]=w[1]X+b[1]
	
	 a[1]=σ(z[1])a[1]=σ(z[1])

![神经网络的表示](https://ws1.sinaimg.cn/large/82e16446ly1fjug7o5ldfj20fh0e8my3.jpg)
进过隐藏层后进入输出层，又有:

	 z[2]=w[2]a[1]+b[2]z[2]=w[2]a[1]+b[2]
	
	 a[2]=σ(z[2])a[2]=σ(z[2])

可以发现，在一个的共有l层，且第l层有n[l]n[l]个节点的神经网络中，参数矩阵 w[l]w[l]的大小为n[l]n[l]*n[l−1]n[l−1]，b[l]b[l]的大小为n[l]n[l]*1。

在losgistic回归，通常将参数初始化为零。而在神经网络中，通常将参数w进行随机初始化，参数b则初始化为0。此外，除w、b外的各种参数，如学习率αα、神经网络的层数l，第l层包含的节点数n[l]n[l]及隐藏层中用的哪种激活函数，都称为**超参数（Hyper Parameters）**，因为它们的值决定了参数w、b最后的值。

### 激活函数

建立一个神经网络时，需要关心的一个问题是，在每个不同的独立层中应当采用哪种激活函数。Logistic回归中，一直采用sigmoid函数作为激活函数，其实还有一些更好的选择。

**tanh函数（Hyperbolic Tangent Function，双曲正切函数） **几乎总比sigmoid函数的效果更好，它的表达式为：

	 tanh(z)=ez−e−zez+e−ztanh(z)=ez−e−zez+e−z

函数图像：
![tanh函数](https://ws1.sinaimg.cn/large/82e16446ly1fjuhv55rcsj208w05m744.jpg)
tanh函数其实是sigmoid函数的移位版本。对于隐藏单元，选用tanh函数作为激活函数的话，效果总比sigmoid函数好，因为tanh函数的值在-1到1之间，最后输出的结果的平均值更趋近于0，而不是采用sigmoid函数时的0.5，这实际上可以使得下一层的学习变得更加轻松。对于二分类问题，为确保输出在0到1之间，将仍然采用sigmiod函数作为输出的激活函数。

然而sigmoid函数和tanh函数都具有的缺点之一是，在z接近无穷大或无穷小时，这两个函数的导数也就是梯度变得非常小，此时梯度下降的速度也会变得非常慢。

线性修正单元，也就是上面举例解释什么是神经网络时用到的ReLU函数也是机器学习中常用到的激活函数之一，它的表达式为：

	 g(z)=max(0,z)={0,z,(z ≤ 0)(z > 0)g(z)=max(0,z)={0,(z ≤ 0)z,(z > 0)

函数图像：
![ReLU函数](https://ws1.sinaimg.cn/large/82e16446ly1fjujlni8kyj20dp07d744.jpg)
当z大于0时是，ReLu函数的导数一直为1，所以采用ReLU函数作为激活函数时，随机梯度下降的收敛速度会比sigmoid及tanh快得多，但负数轴的数据都丢失了。

此外，还有另一个版本的ReLU函数，称为**Leaky-ReLU**，其表达式为：

 g(z)=max(0,z)={αz,z,(z ≤ 0)(z > 0)g(z)=max(0,z)={αz,(z ≤ 0)z,(z > 0)

函数图像：
![Leaky-ReLU](https://ws1.sinaimg.cn/large/82e16446ly1fjujlnn9tgj20de07m3ye.jpg)
其中αα是一个很小的常数，用来保留一部非负数轴的值。

### 前向传播和反向传播

如图，通过输入样本xx及参数w[1]w[1]、b[1]b[1]到隐藏层，求得z[1]z[1]，进而求得a[1]a[1]；再将参数w[2]w[2]、b[2]b[2]和a[1]a[1]一起输入输出层求得z[2]z[2]，进而求得a[2]a[2]，最后得到损失函数(a[2],y)L(a[2],y)，这样一个从前往后递进传播的过程，就称为**前向传播（Forward Propagation）**。
![前向传播](https://ws1.sinaimg.cn/large/82e16446ly1fjupgnkcbtj20r4089q32.jpg)
前向传播过程中：

	 z[1]=w[1]TX+b[1]z[1]=w[1]TX+b[1]
	
	 a[1]=g(z[1])a[1]=g(z[1])
	
	 z[2]=w[2]Ta[1]+b[2]z[2]=w[2]Ta[1]+b[2]
	
	 a[2]=σ(z[2])=sigmoid(z[2])a[2]=σ(z[2])=sigmoid(z[2])
	
	 (a[2],y)=−(ylog a[2]+(1−y)log(1−a[2]))L(a[2],y)=−(ylog a[2]+(1−y)log(1−a[2]))

在训练过程中，经过前向传播后得到的最终结果跟训练样本的真实值总是存在一定误差，这个误差便是损失函数。想要减小这个误差，当前应用最广的一个算法便是梯度下降，于是用损失函数，从后往前，依次求各个参数的偏导，这就是所谓的**反向传播（Back Propagation）**，一般简称这种算法为**BP算法**。
![反向传播](https://ws1.sinaimg.cn/large/82e16446ly1fjupevz9aaj20r00aqt99.jpg)
sigmoid函数的导数为：

	 a[2]′=sigmoid(z[2])′=∂a[2]∂z[2]=a[2](1−a[2])a[2]′=sigmoid(z[2])′=∂a[2]∂z[2]=a[2](1−a[2])

由复合函数求导中的链式法则，反向传播过程中：

			 da[2]=∂(a[2],y)∂a[2]=−ya[2]+1−y1−a[2]da[2]=∂L(a[2],y)∂a[2]=−ya[2]+1−y1−a[2]
	
	 dz[2]=∂(a[2],y)∂a[2]⋅∂a[2]∂z[2]=a[2]−ydz[2]=∂L(a[2],y)∂a[2]⋅∂a[2]∂z[2]=a[2]−y
	
	 dw[2]=∂(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂w[2]=dz[2]⋅a[1]Tdw[2]=∂L(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂w[2]=dz[2]⋅a[1]T
	
	 db[2]=∂(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂b[2]=dz[2]db[2]=∂L(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂b[2]=dz[2]
	
	 da[1]=∂(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂a[1]=dz[2]⋅w[2]da[1]=∂L(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂a[1]=dz[2]⋅w[2]
	
	 dz[1]=∂(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂a[1]⋅∂a[1]∂z[1]=dz[2]⋅w[2]∗g[1]′(z[1])dz[1]=∂L(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂a[1]⋅∂a[1]∂z[1]=dz[2]⋅w[2]∗g[1]′(z[1])
	
	 dw[1]=∂(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂a[1]⋅∂a[1]∂z[1]⋅∂z[1]∂w[1]=dz[1]⋅XTdw[1]=∂L(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂a[1]⋅∂a[1]∂z[1]⋅∂z[1]∂w[1]=dz[1]⋅XT
	
	 db[1]=∂(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂a[1]⋅∂a[1]∂z[1]⋅∂z[1]∂b[1]=dz[1]db[1]=∂L(a[2],y)∂a[2]⋅∂a[2]∂z[2]⋅∂z[2]∂a[1]⋅∂a[1]∂z[1]⋅∂z[1]∂b[1]=dz[1]

这便是反向传播的整个推导过程。

在具体的算法实现过程中，还是需要采用logistic回归中用到梯度下降的方法，将各个参数进行向量化、取平均值，不断进行更新。

### 深层神经网络

深层神经网络含有多个隐藏层，构建方法如前面所述，训练时根据实际情况选择激活函数，进行前向传播获得成本函数进而采用BP算法，进行反向传播，梯度下降缩小损失值。

拥有多个隐藏层的深层神经网络能更好得解决一些问题。如图，例如利用神经网络建立一个人脸识别系统，输入一张人脸照片，深度神经网络的第一层可以是一个特征探测器，它负责寻找照片里的边缘方向，**卷积神经网络（Convolutional Neural Networks，CNN）**专门用来做这种识别。
![深层神经网络](https://ws1.sinaimg.cn/large/82e16446ly1fjxyv40x0kj20kj09l419.jpg)
深层神经网络的第二层可以去探测照片中组成面部的各个特征部分，之后一层可以根据前面获得的特征识别不同的脸型的等等。这样就可以将这个深层神经网络的前几层当做几个简单的探测函数，之后将这几层结合在一起，组成更为复杂的学习函数。从小的细节入手，一步步建立更大更复杂的模型，就需要建立深层神经网络来实现。

### Python实现

#### 花瓣颜色分类器的实现

新用到的Python包有：

*   [sklearn](http://scikit-learn.org/)提供了数种聚类算法，为数据挖掘和数据分析提供了简单有效的工具。

1.生成样本数据

```Python
 
 #生成样本数据

 def load_planar_dataset():

 m = 400 #总样本数

 N = int(m/2) #每种样本数

 D = 2 #维数

 a = 4 #花瓣延伸的最大长度

 X = np.zeros((m,D)) #初始化样本矩阵

 Y = np.zeros((m,1), dtype='uint8') #初始化标签矩阵，0为红色，1为蓝色

 #随机分配样本坐标，使样本组成一朵花形

 for j in range(2):

 ix = range(N*j,N*(j+1))

 t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 #角度

 r = a*np.sin(4*t) + np.random.randn(N)*0.2 #半径

 X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]

 Y[ix] = j

 X = X.T

 Y = Y.T

 plt.scatter(X[0,:], X[1,:], c=Y, s=40, cmap=plt.cm.Spectral);

 return X,Y
```

这些样本数据组成一朵杂带红蓝颜色的花朵图片：
![样本数据](https://ws1.sinaimg.cn/large/82e16446ly1fjydxfg5vij20ae070t9g.jpg)

2.采用logistic回归进行分类
运用sklearn中的提供logistic回归模型，对花朵样本进行分类

```Python

 #生成分类器的边界

 def plot_decision_boundary(model, X, y):

 x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1

 y_min, y_max = X[1,:].min() - 1, X[0,:].max() + 1

 h = 0.01

 xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

 Z = model(np.c_[xx.ravel(), yy.ravel()])

 Z = Z.reshape(xx.shape)

 plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

 plt.ylabel('x2')

 plt.xlabel('x1')

 plt.scatter(X[0,:], X[1,:], c=y, cmap=plt.cm.Spectral)

 #采用sklearn中的logistic回归模型进行分类

 clf = sklearn.linear_model.LogisticRegressionCV() #初始化分类器

 clf.fit(X.T,Y.T.ravel()) #数据拟合

 plot_decision_boundary(lambda x: clf.predict(x), X, Y)

 plt.title("Logistic Regression")

 LR_predictions = clf.predict(X.T)

 print('logistic回归的精确度：%d' % float((np.dot(Y,LR_predictions) + np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) + "%")

 ```

logistic回归分类后的效果为：
![logistic回归分类](https://ws1.sinaimg.cn/large/82e16446ly1fjye29jscaj20as07q0tb.jpg)
且logistic回归分类的准确度只有47%。

3.建立神经网络模型

```Python

 #部分代码此处省略，详细代码见参考资料-Github

 #建立神经网络模型

 def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):

 np.random.seed(3)

 n_x = layer_sizes(X, Y)[0]

 n_y = layer_sizes(X, Y)[2]

 parameters = initialize_parameters(n_x, n_h, n_y);

 W1 = parameters["W1"]

 b1 = parameters["b1"]

 W2 = parameters["W2"]

 b2 = parameters["b2"]

 for i in range(0, num_iterations):

 A2, cache = forward_propagation(X, parameters)

 cost = compute_cost(A2, Y, parameters)

 grads = backward_propagation(parameters, cache, X, Y)

 parameters = update_parameters(parameters, grads)

 if print_cost and i % 1000 == 0:

 print("循环%i次后的成本: %f" %(i, cost))

 return parameters

 #预测结果

 def predict(parameters, X):

 A2, cache= forward_propagation(X, parameters)

 predictions = (A2 > 0.5)

 return predictions

 #将数据输入神经网络模型

 parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

 predictions = predict(parameters, X)

 print ('准确度: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%') #打印精确度

 plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

 plt.title("Decision Boundary for hidden layer size " + str(4))

```

得到的结果为：
![成本](https://ws1.sinaimg.cn/large/82e16446ly1fjyealokafj20dn05idfx.jpg)
![分类结果](https://ws1.sinaimg.cn/large/82e16446ly1fjyeanai0hj20as07qt9a.jpg)

采用一层隐含神经网络进行分类，准确度达到了90%。

4.不同隐藏层节点数下分类

```Python

 #不同隐藏层节点数下分类效果

 plt.figure(figsize=(16, 32))

 hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50, 100]

 for i, n_h in enumerate(hidden_layer_sizes):

 plt.subplot(5, 2, i+1)

 plt.title('Hidden Layer of size %d' % n_h)

 parameters = nn_model(X, Y, n_h, num_iterations = 5000)

 plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

 predictions = predict(parameters, X)

 accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)

 print ("节点数为{}时的分类准确度为 : {} %".format(n_h, accuracy))

```

得到的结果为：
![不同隐藏层节点数下准确度](https://ws1.sinaimg.cn/large/82e16446ly1fjyeff4m9vj20cd04bdfy.jpg)
![不同隐藏层节点数下分类结果](https://ws1.sinaimg.cn/large/82e16446ly1fjyedehh20j20qa141jy6.jpg)

可以发现，随着隐藏层节点数的增加，分类的准确度进一步提高，然而隐藏层节点数过多时，过度学习反而使分类的准确度下降。

#### 基于深层网络的猫识别器的实现

之前使用logistic回归实现过一个猫识别器，最终用测试集测试最高能达到的准确率有70%，现在用一个4层神经网络来实现。

1.L层神经网络参数初始化

```Python

 #初始化参数

 def initialize_parameters_deep(layer_dims):

 np.random.seed(1)

 parameters = {}

 L = len(layer_dims)

 for l in range(1, L):

 parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #初始化为随机值

 parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) #初始化为0

 return parameters

 ```

2.前向传播和后向传播

```Python

 #部分代码此处省略，详细代码见参考资料-Github

 #L层神经网络模型的前向传播

 def L_model_forward(X, parameters):

 caches = []

 A = X

 L = len(parameters) // 2

 for l in range(1, L):

 A_prev = A

 A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")

 caches.append(cache)

 AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")

 caches.append(cache)

 assert(AL.shape == (1,X.shape[1]))

 return AL, caches

 #L层神经网络模型的反向传播

 def L_model_backward(AL, Y, caches):

 grads = {}

 L = len(caches)

 m = AL.shape[1]

 Y = Y.reshape(AL.shape)

 dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

 current_cache = caches[L-1]

 grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

 for l in reversed(range(L-1)):

 current_cache = caches[l]

 dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")

 grads["dA" + str(l + 1)] = dA_prev_temp

 grads["dW" + str(l + 1)] = dW_temp

 grads["db" + str(l + 1)] = db_temp

 return grads

```

3.L层神经网络模型

```Python

 #L层神经网络模型

 def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

 np.random.seed(1)

 costs = []

 parameters = initialize_parameters_deep(layers_dims)

 for i in range(0, num_iterations):

 AL, caches = L_model_forward(X, parameters)

 cost = compute_cost(AL, Y)

 grads = L_model_backward(AL, Y, caches)

 parameters = update_parameters(parameters, grads, learning_rate)

 if print_cost and i % 100 == 0:

 print ("循环%i次后的成本值: %f" %(i, cost))

 if print_cost and i % 100 == 0:

 costs.append(cost)

 plt.plot(np.squeeze(costs))

 plt.ylabel('cost')

 plt.xlabel('iterations (per tens)')

 plt.title("Learning rate =" + str(learning_rate))

 plt.show()

 return parameters

 ```

4.输入数据，得出结果

```Python

 train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

 m_train = train_x_orig.shape[0] #训练集中样本个数

 m_test = test_x_orig.shape[0] #测试集总样本个数

 num_px = test_x_orig.shape[1] #图片的像素大小

 train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T #原始训练集的设为（12288*209）

 test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T #原始测试集设为（12288*50）

 train_x = train_x_flatten/255. #将训练集矩阵标准化

 test_x = test_x_flatten/255. #将测试集矩阵标准化

 layers_dims = [12288, 20, 7, 5, 1]

 parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

 pred_train = predict(train_x, train_y, parameters)

 pred_test = predict(test_x, test_y, parameters)

```

得到的结果为：
![成本值](https://ws1.sinaimg.cn/large/82e16446ly1fjzarivaoej20g4097glx.jpg)
![成本变化曲线](https://ws1.sinaimg.cn/large/82e16446ly1fjzarhqjegj20au07qq2z.jpg)
样本集预测准确度: 0.985645933014
测试集预测准确度: 0.8

### 参考资料

1.  [吴恩达-神经网络与深度学习-网易云课堂](http://mooc.study.163.com/learn/deeplearning_ai-2001281002)
2.  [Andrew Ng-Neural Networks and Deep Learning-Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/)
3.  [deeplearning.ai](https://www.deeplearning.ai/)
4.  [代码及课件资料-GitHub](https://github.com/BinWeber/Deep-Learning)

注：本文涉及的图片及资料均整理翻译自Andrew Ng的Neural Networks and Deep Learning课程，版权归其所有。翻译整理水平有限，如有不妥的地方欢迎指出。

更新历史：

*   2017.09.28 完成初稿