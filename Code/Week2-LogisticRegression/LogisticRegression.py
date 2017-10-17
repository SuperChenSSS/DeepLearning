#logistic_regression

#导入用到的包
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

#导入数据
def load_dataset():
    train_dataset = h5py.File("train_cat.h5", "r")  # 读取训练数据，共209张图片
    test_dataset = h5py.File("test_cat.h5", "r")  # 读取测试数据，共50张图片

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 原始训练集（209*64*64*3）
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 原始训练集的标签集（y=0非猫,y=1是猫）（209*1）

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 原始测试集（50*64*64*3
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 原始测试集的标签集（y=0非猫,y=1是猫）（50*1）

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 原始训练集的标签集设为（1*209）
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # 原始测试集的标签集设为（1*50）

    classes = np.array(test_dataset["list_classes"][:])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#显示图片
def image_show(index,dataset):
    index = index
    if dataset == "train":
        plt.imshow(train_set_x_orig[index])
        print("y = " + str(train_set_y[:,index]) + ",它是一张" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " 图片。")
    elif dataset = "test":
        plt.imshow(test_set_x_orig[index])
        print("y = " + str(test_set_y[:,index]) + ",它是一张" + classes[np.squeeze(test_set_y[:,index])].decode("utf-8") + " 图片。")

#sigmoid函数
def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s

#初始化参数w,b
def initialize_with_zeros(dim):
    w = np.zeros((dim,1)) #w为一个dim*1矩阵
    b = 0
    return w,b

#计算Y_hat,成本函数J以及dw,db
def propagate(w,b,X,Y):
    m = X.shape[1] #样本个数
    Y_hat = sigmoid(np.dot(w.T,X)+b)
    cost = -(np.sum(np.dot(Y,np.log(Y_hat).T)+np.dot((1-Y),np.log(1-Y_hat).T)))/m #成本函数

    dw = (np.dot(X,(Y_hat-Y).T))/m
    db = (np.sum(Y_hat-Y))/m

    cost = np.squeeze(cost) #压缩维度
    grads = {"dw": dw,
             "db": db} #梯度

    return grads,cost

#梯度下降找出最优解
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False) #num_iterations-梯度下降次数 learning_rate-学习率，即参数a
    costs = [] #记录成本值

    for i in range(num_iterations): #循环进行梯度下降
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0: #每100次记录一次成本值
            costs.append(cost)

        if print_cost and i % 100 == 0: #打印成本值
            print("循环%i次后的成本值：%f" %(i,cost))

    params = {"w": w,
              "b": b} #最终参数值

    grads = {"dw": dw,
             "db": db}#最终梯度值

    return params,grads,costs

#预测出结果
def predict(w,b,X):
    m = X.shape[1] #样本个数
