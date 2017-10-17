# 深度学习：Logistic 回归


**深度学习（Deep Learning）**是**机器学习（Machine Learning）**的一大分支，它试图使用包含复杂结构或由多重非线性变换构成的多个处理层对数据进行高层抽象的算法。

**逻辑回归（Logistic Regression，也译作”对数几率回归“）**是离散选择法模型之一，属于多重变量分析范畴，是社会学、生物统计学、临床、数量心理学、计量经济学、市场营销等统计实证分析的常用方法。

<a id="more"></a>
### 符号约定

逻辑回归问题是一类**二分类（Binary Classification）**问题，给定一些输入，输出的结果是离散值。

例如：为了训练一个猫识别器，输入一张图片表示为特征向量x，并预测图片是否为猫，输出y为1（是）或0（不是）。
![猫分类器](https://ws1.sinaimg.cn/large/82e16446ly1fjers3qr63j20r706e41q.jpg)
我们称图片为非结构化数据，但在计算机中，一张图片以RGB方式编码时，它是以Red、Green、Blue三基色组成一个矩阵的方式进行储存的，这三个矩阵的大小和图片的大小相同 ，如图中每张猫图大小为64 pixels*64 pixels的，那么三个矩阵中每个矩阵的大小即为64*64。

单元格中代表的像素值将用来组成一个N维的特征向量。在**模式识别（Pattern Recognition）**和机器学习中，一个特征向量用来表示一个对象。这个问题中，这个对象为猫或者非猫。

为了组成一个特征向量x，将每一种颜色的像素值进行拆分重塑，最终形成的特征向量x的维数为nxnx = 64*64*3 = 12288。
![特征向量x](https://ws1.sinaimg.cn/large/82e16446ly1fjet6mny7nj20fd05laa2.jpg)
一个训练样本由一对(x,y)进行表示，其中x为nxnx维特征向量，y是这个特征向量的标签，值为0或1。m个训练样本对将被表示为：

 (x(1),y(1)),(x(2),y(2)),…,(x(m),y(m))(x(1),y(1)),(x(2),y(2)),…,(x(m),y(m))

定义矩阵X、Y，将输入的训练集中的x(1)x(1),x(2)x(2)等以及y(1)y(1),y(2)y(2)等分别堆叠在矩阵的列向量中：

 X=[x(1),x(2),…,x(m)]X=[x(1),x(2),…,x(m)]

 Y=[y(1),y(2),…,y(m)]Y=[y(1),y(2),…,y(m)]

其中X为nxnx*m矩阵，Y为1*m矩阵。

Python中即X.shape=(nxnx,m)，Y.shape=(1,m)。

### Logistic 回归

Logistic 回归是一种用于解决**监督学习（Supervised Learning）**问题的学习算法，其输出y的值为0或1。Logistic回归的目的是使训练数据与其预测值之间的误差最小化。

下面以Cat vs No-cat为例：
给定以一个nxnx维特征向量x表示的一张图片，这个算法将估计这张图中存在一只猫的概率，即y=1的概率：

 给定 x,计算ŷ =P(y=1|x),且 0≤ŷ ≤1给定 x,计算y^=P(y=1|x),且 0≤y^≤1

我们希望能有一个函数，能够表示出ŷ y^，如果进行最简单的线性拟合的话，规定一个nxnx维向量w和一个值b作为参数，可得到：

 ŷ =wTX+by^=wTX+b

但由于ŷ y^为一个概率值，取值范围为[0,1]，简单地进行线性拟合，得出的ŷ y^可能非常大，还可能为负值。这时，便需要一个**sigmoid函数**来对它的值域进行约束，sigmoid函数的表达式为：

 σ(z)=11+e−zσ(z)=11+e−z

其函数图像为：
![sigmoid函数 ](https://ws1.sinaimg.cn/large/82e16446ly1fjfonyqt7sj20lb08mab8.jpg)
由函数图像可知，sigmoid函数有几个很好的性质：

*   当z趋近于正无穷大时，σ(z) = 1
*   当z趋近于负无穷大时，σ(z) = 0
*   当z = 0时，σ(z) = 0.5

所以可以用sigmoid函数来约束ŷ y^的值域，此时：

 ŷ =σ(wTX+b)=11+e−(wTX+b)y^=σ(wTX+b)=11+e−(wTX+b)

#### 成本函数

为了训练logistic回归模型中的参数w和b，使得我们的模型输出值ŷ y^与真实值y尽可能基本一致，即尽可能准确地判断一张图是否为猫，我们需要定义一个**成本函数（Cost Function）**作为衡量的标准。

用**损失函数（Loss Function）**来衡量预测值（ŷ (i)y^(i)）与真实值（y(i)y(i)）之间的差异，换句话说，损失函数用来计算单个训练样本的错误。平方误差（Square Loss）是一种常用的损失函数：

 (ŷ ,y)=12(ŷ −y)2L(y^,y)=12(y^−y)2

但在logistic回归中一般不使用这个损失函数，因为在训练参数过程中，使用这个损失函数将得到一个非凸函数，最终将存在很多局部最优解，这种情况下使用**梯度下降（Gradient Descent）**法无法找到最优解。所以在logistic回归中，一般采用log函数：

 (ŷ ,y)=−(ylog ŷ +(1−y)log(1−ŷ ))L(y^,y)=−(ylog y^+(1−y)log(1−y^))

log损失函数有如下性质：

*   当y(i)=1y(i)=1时，(ŷ (i),y(i))=−ylog(ŷ (i))L(y^(i),y(i))=−ylog(y^(i))
*   当y(i)=0y(i)=0时，(ŷ (i),y(i))=−ylog(1−ŷ (i))L(y^(i),y(i))=−ylog(1−y^(i))

成本函数是整个训练集的损失函数的平均值：

 J(w,b)=1m∑i=1m(ŷ (i),y(i))J(w,b)=1m∑i=1mL(y^(i),y(i))

我们要找到参数w和b，使这个成本函数的值最小化。

#### 梯度下降

标量场中某一点上的梯度指向标量场增长最快的方向，梯度的长度是这个最大的变化率。
![J(w,b)空间图像](https://ws1.sinaimg.cn/large/82e16446ly1fjfyo6aitfj20eh079ab5.jpg)
在空间坐标中以w，b为轴画出损失函数J(w,b)的三维图像，可知这个函数为一个凸函数。为了找到合适的参数，先将w和b赋一个初始值，正如图中的小红点。在losgistic回归中，几乎任何初始化方法都有效，通常将参数初始化为零。随机初始化也起作用，但通常不会在losgistic回归中这样做，因为这个成本函数是凸的，无论初始化的值是多少，总会到达同一个点或大致相同的点。梯度下降就是从起始点开始，试图在最陡峭的下降方向下坡，以便尽可能快地下坡到达最低点，这个下坡的方向便是此点的梯度值。
![J(w,b)二维图像](https://ws1.sinaimg.cn/large/82e16446ly1fjfz0ohz1dj20eh076wep.jpg)
在二维图像中来看，顺着导数的方向，下降速度最快，用数学公式表达即是：

 w:=w−α∂J(w,b)∂ww:=w−α∂J(w,b)∂w

 b:=b−α∂J(w,b)∂bb:=b−α∂J(w,b)∂b

其中的“:=”意思为赋值，αα为学习率，通常为一个小于1的数，用来控制梯度下降过程中每一次移动的规格，相当于迈的步子大小。αα的不宜太小也不宜过大：太小会使迭代次数增加，容易陷入局部最优解；太大容易错过最优解。

### Python实现

使用Python编写一个logistic回归分类器来识别猫，以此来了解如何使用神经网络的思维方式来进行这项任务，识别过程如图：
![识别过程示意图](https://ws1.sinaimg.cn/large/82e16446ly1fjh20slwq7j20ss0mc43f.jpg)

#### 实现过程

其中用到的Python包有：

*   [numpy](http://www.numpy.org/)是使用Python进行科学计算的基础包。
*   [matplotlib](http://matplotlib.org/)是Python中著名的绘图库。
*   [h5py](http://www.h5py.org/)在Python提供读取HDF5二进制数据格式文件的接口，本次的训练及测试图片集是以HDF5储存的。
*   [PIL](http://www.pythonware.com/products/pil/)(Python Image Library)为Python提供图像处理功能。
*   [scipy](https://www.scipy.org/)基于NumPy来做高等数学、信号处理、优化、统计和许多其它科学任务的拓展库。
    几个Python包的安装及基本使用方法详见官网。

1.导入要用到的所有包

 #导入用到的包

 	import numpy as np
 	import matplotlib.pyplot as plt
 	import h5py
 	import scipy
 	from PIL import Image
 	from scipy import ndimage
 
2.导入数据

```Python
 #导入数据

 def load_dataset():
	 	train_dataset = h5py.File("train_cat.h5","r") #读取训练数据，共209张图片
	 	test_dataset = h5py.File("test_cat.h5", "r") #读取测试数据，共50张图片
	 	train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #原始训练集（209*64*64*3）
	 	train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #原始训练集的标签集（y=0非猫,y=1是猫）（209*1）
	 	test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #原始测试集（50*64*64*3) 	
	 	test_set_y_orig = np.array(test_dataset["test_set_y"][:]) #原始测试集的标签集（y=0非猫,y=1是猫）（50*1）
	 	train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0])) #原始训练集的标签集设为（1*209）
	 	test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0])) #原始测试集的标签集设为（1*50）
	 	classes = np.array(test_dataset["list_classes"][:])
	 	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
```
 

需要说明的是，本次的训练及测试图片集是以HDF5格式储存的,train_cat.h5、test_cat.h5文件打开后结构如下：

![h5文件结构](https://ws1.sinaimg.cn/large/82e16446ly1fjh1zdo2tij20h70dh0ur.jpg)

另外，也可以调用以下方法来查看训练集或测试集中的图片：

```Python
 	#显示图片
 	def image_show(index,dataset):

 	index = index
 	if dataset == "train":
		plt.imshow(train_set_x_orig[index])
	 	print ("y = " + str(train_set_y[:, index]) + ", 它是一张" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' 图片。")
	 elif dataset == "test":
	 	plt.imshow(test_set_x_orig[index])
	 	print ("y = " + str(test_set_y[:, index]) + ", 它是一张" + classes[np.squeeze(test_set_y[:, index])].decode("utf-8") + "' 图片。")
```
 

3.sigmoid函数

```
	#sigmoid函数
	
	def sigmoid(z):
 		s = 1/(1+np.exp(-z))
 	return s
```

4.初始化参数w，b

```Python
	 #初始化参数w,b
	
	 def initialize_with_zeros(dim):
	
	 w = np.zeros((dim,1)) #w为一个dim*1矩阵
	
	 b = 0
	
	 return w, b
```
 

5.计算Y_hat,成本函数J以及dw，db
 
```Python
 	#计算Y_hat,成本函数J以及dw，db

 	def propagate(w, b, X, Y):

	 m = X.shape[1] #样本个数
	 Y_hat = sigmoid(np.dot(w.T,X)+b)
	 cost = -(np.sum(np.dot(Y,np.log(Y_hat).T)+np.dot((1-Y),np.log(1-Y_hat).T)))/m #成本函数
	 dw = (np.dot(X,(Y_hat-Y).T))/m
	 db = (np.sum(Y_hat-Y))/m
	 cost = np.squeeze(cost) #压缩维度
	 grads = {"dw": dw,
	 "db": db} #梯度
	 return grads, cost
```
 

6.梯度下降找出最优解

```Python
 	#梯度下降找出最优解
 
 	def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):#num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ

	 costs = [] #记录成本值	
	 for i in range(num_iterations): #循环进行梯度下降
	 grads, cost = propagate(w,b,X,Y)
	 dw = grads["dw"]
	 db = grads["db"]
	 w = w - learning_rate*dw
	 b = b - learning_rate*db
	 if i % 100 == 0: #每100次记录一次成本值
	 	costs.append(cost)
	 if print_cost and i % 100 == 0: #打印成本值
	 	print ("循环%i次后的成本值: %f" %(i, cost))
	 params = {"w": w,
	 "b": b} #最终参数值
	 grads = {"dw": dw,
	 "db": db}#最终梯度值
	 return params, grads, costs
```
 

7.得出预测结果

```Python
	 #预测出结果
	
	 def predict(w, b, X):
	
	 m = X.shape[1] #样本个数
	
	 Y_prediction = np.zeros((1,m)) #初始化预测输出
	
	 w = w.reshape(X.shape[0], 1) #转置参数向量w
	
	 Y_hat = sigmoid(np.dot(w.T,X)+b) #最终得到的参数代入方程
	
	 for i in range(Y_hat.shape[1]):
	
	 if Y_hat[:,i]>0.5:
	
	 Y_prediction[:,i] = 1
	
	 else:
	
	 Y_prediction[:,i] = 0
	
	 return Y_prediction
```
 

8.建立整个预测模型

```Python
	 #建立整个预测模型
	
	 def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False): #num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ
	
	 w, b = initialize_with_zeros(X_train.shape[0]) #初始化参数w，b
	
	 parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost) #梯度下降找到最优参数
	
	 w = parameters["w"]
	
	 b = parameters["b"]
	
	 Y_prediction_train = predict(w, b, X_train) #训练集的预测结果
	
	 Y_prediction_test = predict(w, b, X_test) #测试集的预测结果
	
	 train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100 #训练集识别准确度
	
	 test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100 #测试集识别准确度
	
	 print("训练集识别准确度: {} %".format(train_accuracy))
	
	 print("测试集识别准确度: {} %".format(test_accuracy))
	
	 d = {"costs": costs,
	
	 "Y_prediction_test": Y_prediction_test,
	
	 "Y_prediction_train" : Y_prediction_train,
	
	 "w" : w,
	
	 "b" : b,
	
	 "learning_rate" : learning_rate,
	
	 "num_iterations": num_iterations}
	
	 return d
```
 

9.初始化样本，输入模型，得出结果

```Python
	 #初始化数据
	
	 train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
	
	 m_train = train_set_x_orig.shape[0] #训练集中样本个数
	
	 m_test = test_set_x_orig.shape[0] #测试集总样本个数
	
	 num_px = test_set_x_orig.shape[1] #图片的像素大小
	
	 train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T #原始训练集的设为（12288*209）
	
	 test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T #原始测试集设为（12288*50）
	
	 train_set_x = train_set_x_flatten/255. #将训练集矩阵标准化
	
	 test_set_x = test_set_x_flatten/255. #将测试集矩阵标准化
	
	 d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
```
 

#### 结果分析

运行程序最终得到的结果为：
![结果](https://ws1.sinaimg.cn/large/82e16446ly1fjh33bb9juj20em0atjri.jpg)
训练集识别准确率接近100％，测试集的准确率有70％。由于训练使用的小数据集，而且logistic回归是线性分类器，所以这个结果对于这个简单的模型实际上还是不错。

使用mathplotlib画出学习曲线：

```Python
	 # 画出学习曲线
	
	 costs = np.squeeze(d['costs'])
	
	 plt.plot(costs)
	
	 plt.ylabel('cost')
	
	 plt.xlabel('iterations (per hundreds)')
	
	 plt.title("Learning rate =" + str(d["learning_rate"]))
	
	 plt.show()
```
 

![学习曲线](https://ws1.sinaimg.cn/large/82e16446ly1fjh3i9cu8vj20at07q3yj.jpg)

学习率不同时的学习曲线：


	 learning_rates = [0.01, 0.001, 0.0001]
	
	 models = {}
	
	 for i in learning_rates:
	
	 print ("学习率: " + str(i))
	
	 models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
	
	 print ('\n' + "-------------------------------------------------------" + '\n')
	
	 for i in learning_rates:
	
	 plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
	
	 plt.ylabel('cost')
	
	 plt.xlabel('iterations')
	
	 legend = plt.legend(loc='upper center', shadow=True)
	
	 frame = legend.get_frame()
	
	 frame.set_facecolor('0.90')
	
	 plt.show()

 

![几个不同学习率下的学习曲线](https://ws1.sinaimg.cn/large/82e16446ly1fjh3t7y41xj20at07e3yo.jpg)
说明不同的学习率会带来不同的成本，从而产生不同的预测结果。

### 参考资料

1.  [吴恩达-神经网络与深度学习-网易云课堂](http://mooc.study.163.com/learn/deeplearning_ai-2001281002)
2.  [Andrew Ng-Neural Networks and Deep Learning-Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/)
3.  [deeplearning.ai](https://www.deeplearning.ai/)
4.  [代码及课件资料-GitHub](https://github.com/BinWeber/Deep-Learning)

注：本文涉及的图片及资料均整理翻译自Andrew Ng的Neural Networks and Deep Learning课程，版权归其所有。翻译整理水平有限，如有不妥的地方欢迎指出。

更新历史：

*   2017.09.13 完成初稿
*   2017.09.17 修正部分错误