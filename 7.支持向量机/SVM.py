import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#数据处理
#使用iris数据集中的前两类，和前两个特征，线性可分。
#返回特征和标签两个array
def create_date():
    iris=load_iris()
    df=pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label']=iris.target
    df.columns=['sepal_length','sepal_width','petal_length','petal_width','label']
    data=np.array(df.iloc[:100,[0,1,-1]])
    for i in range(len(data)):
        if(data[i,-1]==0):
            data[i,-1]=1
    return data[:,:2],data[:,-1]


#选择数据中的20%作为测试集，设定随机种子
X,y=create_date()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



class SVM:
    def __init__(self,max_iter=100,kernel='linear'):
        self.max_iter=max_iter
        self._kernel=kernel

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        #aloha:构建拉格朗日函数引入的参数
        self.alpha = np.ones(self.m)
        #E:松弛变量
        self.E=np.ones(self.m)
        #C:惩罚参数
        self.C=1.0

    #对xi进行预测，返回预测的结果
    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r



    # 核函数，计算两个样本的核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2

        return 0

    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]



















