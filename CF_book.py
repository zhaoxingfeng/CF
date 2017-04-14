# coding: utf-8
"""
作者：zhaoxingfeng	日期：2017.04.13
功能：协同过滤算法，Collaborative Filtering(CF)，书籍推荐
版本：V1.0
参考文献：
[1] 苍梧.机器学习中的相似性度量[DB/OL].http://www.cnblogs.com/heaad/archive/2011/03/08/1977733.html,2011-03-08.
[2] 赵晨婷,马春娥.探索推荐引擎内部的秘密，第2部分: 深入推荐引擎相关算法-协同过滤[DB/OL].
    https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy2/index.html,2011-03-21.
"""
from __future__ import division
import pandas as pd
import numpy as np

class CF:
    """
    data:字典类型
    func_name:选择距离函数
    k:近邻个数
    count:推荐的物品数量
    """
    def __init__(self, data, func_name='pearson', k=2, count=10):
        self.data = data
        if func_name == 'pearson':
            self.func = self.calcPearson
        elif func_name == 'cosine':
            self.func = self.calcCosine
        self.k = k
        self.count = count

    # 根据对物品的评分计算用户之间的pearson相关系数
    def calcPearson(self, dict1, dict2):
        n = 0
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        for i in dict1:
            if i in dict2:
                n += 1
                sum_xy += dict1[i] * dict2[i]
                sum_x += dict1[i]
                sum_y += dict2[i]
                sum_x2 += dict1[i] ** 2
                sum_y2 += dict2[i] ** 2
        num = sum_xy - sum_x * sum_y / n
        den = ((sum_x2 - pow(sum_x, 2) / n) * (sum_y2 - pow(sum_y, 2) / n)) ** 0.5
        if den == 0:
            return 0
        else:
            return num / den

    # 根据对物品的评分计算用户之间的余弦相关系数
    def calcCosine(self, dict1, dict2):
        vect1, vect2 = [], []
        for i in dict1:
            if i in dict2:
                vect1.append(dict1[i])
                vect2.append(dict2[i])
        num = np.multiply(vect1, vect2)
        den = (sum([x*x for x in vect1]) * sum([y*y for y in vect2])) ** 0.5
        return sum(num) / den

    # 将待求人名和字典中的所有人名比较，得到相似度排序
    def sortNeighbor(self, userID):
        userDict = {}
        for key in self.data:
            if key != userID:
                coeff = self.func(self.data[key], self.data[userID])
                userDict[key] = coeff
        userDict = sorted(userDict.items(), key=lambda x: x[1], reverse=True)
        return userDict

    # 主函数
    def recommend(self, userID):
        # 返回的推荐列表
        recommendDict = {}
        useridDict = self.data[userID]
        neighborName = self.sortNeighbor(userID)
        # k个用户各自的权重，用于累加评分
        weight_sum = sum([neighborName[i][1] for i in range(self.k)])
        for j in range(self.k):
            weight_j = neighborName[j][1] / weight_sum
            neighbor_j = self.data[neighborName[j][0]]
            # 不在所求用户已知的物品列表里边，也不在已经添加了的推荐列表recommendDict里边
            for t in neighbor_j:
                if t not in useridDict:
                    if t not in recommendDict:
                        recommendDict[t] = neighbor_j[t] * weight_j
                    else:
                        recommendDict[t] += neighbor_j[t] * weight_j
        # 对已经建立的推荐物品字典按照评分由高到低进行排序
        recommendDict = sorted(recommendDict.items(), key=lambda x: x[1], reverse=True)
        return recommendDict[:self.count]


if __name__ == "__main__":
    # 数据结构：用户ID，评分，书籍ID
    df = pd.read_csv(r'Books\book.csv', header=None).values
    user_item_dict = {}
    for dt in df:
        user_item_dict.setdefault(dt[0], {})
        user_item_dict[dt[0]][dt[2]] = dt[1]
    model = CF(user_item_dict, 'pearson', 2, 10)
    # 获取2052828用户的推荐书籍列表
    recommend_result = model.recommend('2052828')
    df = pd.DataFrame(recommend_result, columns=['book', 'rating'])
    print(df)
    df.to_csv('recommend_result.csv', index=None)


