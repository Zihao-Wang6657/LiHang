"""
KNN是一个十分直观，简单的算法，根据已有数据中与待测点x最近的K个点
的标签来判断x的标签。
三要素：1.K的选择。2.距离度量的规定。3.分类决策规则
在实际实现中，主要的问题是如何快速找到x的K个最近邻
简单的方法是暴力计算，但效率低下，另一种常见的方法是用kd树保存并检索已有数据点。
"""
from math import sqrt
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#选择鸢尾花数据集
from sklearn.datasets import load_iris
#用于将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split
#用于对元素进行计数的库
from collections import Counter

#load_data
iris = load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['label']=iris.target
df.columns=['sepal_length','sepal_width','petal_length','petal_width','label']
date=np.array(df.iloc[:,[0,1,2,3,4]])
X,y=date[:,:-1],date[:,-1]
#将数据集划分为训练集(80%)和测试集(20%)，设置随机种子保证可复现性
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class KNN:
    def __init__(self,X_train,y_train,n_neighbors=3,p=2):
        """
        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :param n_neighbors: 选定的用于判断的邻居个数，默认为3
        :param p: 选定的距离度量，默认为2，也就是欧氏距离
        """
        self.X_train=X_train
        self.y_train=y_train
        self.n=n_neighbors
        self.p=p

    def predict(self, X):
        """
        暴力计算K近邻
        :param X:待预测的点
        :return: 返回对应的标签
        """
        knn_list=[]
        for i in range(self.n):
            dist=np.linalg.norm(X-self.X_train[i],ord=self.p)
            knn_list.append((dist,self.y_train[i]))

        for i in range(self.n,len(self.X_train)):
            max_index=knn_list.index(max(knn_list,key=lambda x:x[0]))
            dist=np.linalg.norm(X-self.X_train[i],ord=self.p)
            if dist<knn_list[max_index][0]:
                knn_list[max_index]=(dist,self.y_train[i])

        #统计并返回个数最多的标签
        knn_label = [k[-1] for k in knn_list]
        count_pairs = Counter(knn_label)
        #这行代码较为复杂：count_pairs.items()将counter_pairs变为(label,counter)的类型
        #之后，用lamda语句比较counter并排序，最后取[-1]最后一个(也就是counter最大的)的元素，再取label[0]
        max_count_label = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        return max_count_label

    def score(self,X_test,y_test):
        """
        :param X_test:测试集特征
        :param y_test: 测试集标签
        :return: 得分，也就是分类正确率,介于0到1之间
        """
        right_count=0
        for X,y in zip(X_test,y_test):
            label=self.predict(X)
            if label==y:
                right_count+=1
        return right_count/len(X_test)



clf = KNN(X_train, y_train)
print("在鸢尾花数据集上的正确率为：",clf.score(X_test, y_test))





"""
下面提供kd树的实现，kd树是一种二叉树，k代表数据的维度
kd树的思想：对k维超矩形空间不断切分，每次选择一个维度，在这个维度上选择一个切分点，
沿着垂直于维度轴的方向，对整个超矩形空间进行切分，
切分点一般选择空间中的一个数据点，将这个数据点放到结点上，该节点的两个子节点代表切分出的两个子空间。
(当选取中位数对应的点为切分点时，kd树成为平衡而二叉树，但是此时的搜索效率不一定时最优的)
在下一个深度上，选择下一个维度重复上面的切分方式，直到子空间没有数据点存在。
"""


class KdNode(object):
    def __init__(self,dom_elt,split,left,right):
        """
        节点类
        :param dom_elt: 该节点的数据
        :param split: 这个节点分割的维度序号
        :param left: 左子节点
        :param right: 右子节点
        """
        self.dom_elt=dom_elt
        self.split=split
        self.left=left
        self.right=right

class KdTree(object):
    def __init__(self,data):
        k=len(data[0]) #数据维度

        def CreateNode(split,data_set):
            """
            :param split: 要划分的维度对应的序号
            :param data_set: 在这个超矩形空间的全部数据点的集合
            """
            if not data_set:
                return None
            data_set.sort(key=lambda x:x[split])
            split_pos=len(data_set)//2 #取中位数作为分割点，整除
            median=data_set[split_pos]
            next_split=(split+1)% k
            #递归地创建kd树
            return KdNode(
                median,
                split,
                CreateNode(next_split, data_set[:split_pos]),  # 创建左子树
                CreateNode(next_split, data_set[split_pos + 1:]))  # 创建右子树

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点

#KDTree地前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


#对已经建立好的kd树进行搜索，寻找最近样本点
# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result=namedtuple("Result_tuple","nearest_point nearest_dist nodes_visited")

def find_nearest(tree, point):
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"),
                          0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归




