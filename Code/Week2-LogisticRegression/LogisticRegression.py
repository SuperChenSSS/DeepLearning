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
    train_dataset = h5py.File("train_cat.h5","r") #读取训练数据，共209张图片
    test_dataset = h5py.File("test_cat.h5","r") #读取测试数据，共50张图片

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #原始训练集（209*64*64*3）
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #原始训练集的标签集（y=0非猫，y=1是猫）（209*1）

