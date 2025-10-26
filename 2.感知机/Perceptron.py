import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#使用iris数据集

#加载数据
iris=load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df['label']=iris.target
df.columns=['sepal_length','sepal_width','petal_length','petal_width','label']

#取数据集中的两个鸢尾花品种(线性可分)，再取sepal length，sepal width两个特征(使用这两个特征就可以分开这两个品种)。
data=np.array(df.iloc[:100,[0,1,-1]])
X,y=data[:,:-1],data[:,-1]
#将标签转换为感知机模型要求的1，-1形式
y=np.array([1 if i==1 else -1 for i in y.ravel() ])



#感知机算法
class Perceptron:
    def __init__(self):
        self.w=np.ones(len(data[0])-1,dtype=np.float32)
        self.b=0
        self.learning_rate=0.1

    def sign(self,x,w,b):
        y=np.dot(x,w)+b
        return y

    #随机梯度下降
    def fit(self,X_train,y_train):
        is_wrong=False #数据集中还有没有分类错误的点
        while not is_wrong:
            wrong_count=0
            for d in range(len(X_train)):
                X=X_train[d]
                y=y_train[d]
                if y*self.sign(X,self.w,self.b) <= 0:
                    self.w=self.w+self.learning_rate*np.dot(y,X)
                    self.b=self.b+self.learning_rate*y
                    wrong_count +=1
            if wrong_count==0:
                is_wrong=True
        return 'Training Finished!'

    #用来在测试集上运行算法时对算法打分，尚未实现
    def score(self):
        pass


#初始化并训练模型
perceptron = Perceptron()
perceptron.fit(X, y)

#画图
x_points = np.linspace(4, 7, 10)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'o', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'o', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()