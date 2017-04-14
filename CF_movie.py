# coding: utf-8
"""
作者：zhaoxingfeng	日期：2017.04.14
功能：协同过滤算法，Collaborative Filtering(CF)，电影推荐
版本：V1.0
参考文献：
[1] qq_20282263.python实现协同过滤推荐算法.http://m.blog.csdn.net/article/details?id=52692318,2016-09-28.
"""
from __future__ import division
import pandas as pd
import numpy as np
import time

class CF():
    def __init__(self, ratings, movies, func_name='cosine', k=2, count=10):
        self.ratings = ratings
        self.movies = movies
        if func_name == 'cosine':
            self.func = self.calcCosine
        self.k = k
        self.count = count
        self.userDict = {}
        self.ItemUser = {}

    # 将评分ratings转换为userDict和ItemUser
    def dataClean(self):
        for rate in self.ratings:
            temp = [rate[1], rate[2]]
            if rate[0] not in self.userDict:
                self.userDict.setdefault(rate[0], [])
                self.userDict[rate[0]].append(temp)
            else:
                self.userDict[rate[0]].append(temp)
            if rate[1] not in self.ItemUser:
                self.ItemUser.setdefault(rate[1], [])
                self.ItemUser[rate[1]].append(rate[0])
            else:
                self.ItemUser[rate[1]].append(rate[0])

    # 余弦相关系数
    def calcCosine(self, Dict):
        data = np.array(Dict.values())
        vect1, vect2 = data[:, 0], data[:, 1]
        num = sum(np.multiply(vect1, vect2))
        den = (sum([x**2 for x in vect1]) * sum([y**2 for y in vect2])) ** 0.5
        return num / den

    # 找到userID的相邻k个用户
    def getNeighbor(self, userID):
        neighborsName = []
        for i in self.userDict[userID]:
            for j in self.ItemUser[i[0]]:
                if j != userID and j not in neighborsName:
                    neighborsName.append(j)
        neighborsList = []
        for i in neighborsName:
            # 获取userID与某用户的并集，{电影ID, [userID的评分，某用户的评分]}，没有评分记为0
            user_nighbor = {}
            for j in self.userDict[userID]:
                user_nighbor[j[0]] = [j[1], 0]
            for t in self.userDict[i]:
                if t[0] not in user_nighbor:
                    user_nighbor[t[0]] = [0, t[1]]
                else:
                    user_nighbor[t[0]][1] = t[1]
            dist = self.func(user_nighbor)
            neighborsList.append([dist, i])
        neighborsList = sorted(neighborsList, reverse=True)[:self.k]
        return neighborsList

    # 显示推荐列表
    def showResult(self, resultList):
        showList = []
        for result in resultList:
            movieName = self.movies[self.movies[0] == result[0]][[1, 2]].values
            showList.append([movieName[0][0], movieName[0][1], result[1]])
        df = pd.DataFrame(showList, columns=['movieName', 'Type', 'Rate'])
        df.to_csv('movie_recommend.csv', index=None, sep=',')
        print(df)

    # 主函数
    def recommend(self, userID):
        self.dataClean()
        neighborsList = self.getNeighbor(userID)
        # 建立推荐字典
        recommendDict = {}
        weightSum = sum([x[0] for x in neighborsList])
        for neighbor in neighborsList:
            weight = neighbor[0] / weightSum
            neighborMovies = self.userDict[neighbor[1]]
            for movie in neighborMovies:
                if movie[0] not in np.array(self.userDict[userID])[:, 0]:
                    if movie[0] not in recommendDict:
                        recommendDict[movie[0]] = movie[1] * weight
                    else:
                        recommendDict[movie[0]] += movie[1] * weight
        recommendDict = sorted(recommendDict.items(), key=lambda x: x[1], reverse=True)[:self.count]
        return recommendDict

if __name__ == '__main__':
    start = time.time()
    # 读取用户打分文件，电影ID和电影名对应文件
    # 数据结构：ratings{UserID::MovieID::Rating::Timestamp},movies{MovieID::Title::Genres}
    ratings = pd.read_csv(r"MovieLens\ratings.txt", header=None, sep='::', engine='python').values
    movies = pd.read_csv(r"MovieLens\movies.txt", header=None, sep='::', engine='python')
    model = CF(ratings, movies, 'cosine', k=2, count=5)
    recommendList = model.recommend(1)
    model.showResult(recommendList)
    end = time.time()
    print("time consuming = {}s".format(end - start))
