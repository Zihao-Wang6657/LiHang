import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math


#数据处理的函数
def creat_data():
    """
    没有输入
    :return:鸢尾花数据集的特征和标签两个array
    """
    iris = load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label']=iris.target
    df.columns=['sepal_length','sepal_width','petal_length','petal_width','label']
    data=np.array(df)
    return data[:,:-1],data[:,-1]

#处理数据，划分训练集和测试集
X,y=creat_data()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#在朴素贝叶斯中，需要对特征向量的取值进行假设
#对于离散取值的特征，可以直接用频率计数；对于连续(我们使用的鸢尾花数据库中就是)取值的特征，就要进行假设
#在这里，我们假设特征取值符合高斯分布。
class Naive_Bayes:
    def __init__(self):
        self.model=None

    def mean(self,X):
        """
        计算特征的均值
        :return: 返回每个特征的均值
        """
        return sum(X) / float(len(X))


    def stdev(self,X):
        """
        计算并返回特征的标准差(方差)
        :param X: 数据集的特征部分
        :return: 每个特征的标准差
        """
        avg = self.mean(X)
        deviation=X-avg
        squared_deviation=pow(deviation,2)
        sum=np.sum(squared_deviation,axis=0)
        return math.sqrt(sum/len(X))

    def gaussian_probability(self,x,mean,stdev):
        """
        计算一个特征向量的概率密度函数
        :param x: 特征向量
        :param mean: 特征平均值
        :param stdev: 特征标准差
        :return: 返回特征向量的概率密度
        """
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def summarize(self,train_data):
        """
        处理训练集数据
        :param train_data: 训练集数据
        :return:训练集每个特征的均值和标准差
        """
        summaries=[(self.mean(i),self.stdev(i)) for i in zip(*train_data)]
        return summaries


    def fit(self,X,y):
        """
        根据已知的数据计算概率分布
        :param X:训练集数据特征
        :param y:训练集标签
        :return:每个类别的概率分布函数
        """
        labels=list(set(y))
        data={label:[]for label in labels}
        for f ,label in zip(X,y):
            data[label].append(f)
        self.model={
            label:self.summarize(value)
            for label,value in data.items()
        }
        return "train done"

    def calculate_probabilities(self,input_data):
        """
        :param input_data:待预测的特征向量
        :return:计算出的它属于每个类别的似然度
        """
        probabilities={}
        for label,value in self.model.items():
            probabilities[label]=1
            for i in range(len(value)):
                mean,stdev=value[i]
                probabilities[label]*=self.gaussian_probability(
                    input_data[i],mean,stdev
                )
        return probabilities


    def predict(self,X_test):
        """
        对测试集进行预测
        :param X_test: 测试集特征
        :return: 似然度最大的类别
        """
        label=sorted(
            self.calculate_probabilities(X_test).items(),
            key=lambda x : x[-1]
        )[-1][0]
        return label

    def score(self,X_test,y_test):
        right=0
        for X,y in zip(X_test,y_test):
            label=self.predict(X)
            if label==y:
                right+=1
        return right/float(len(X_test))


#主函数
model=Naive_Bayes()
model.fit(X_train,y_train)
print("朴素贝叶斯在鸢尾花数据集上的正确率：",model.score(X_test,y_test))
