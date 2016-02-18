#coding=utf-8
'''
@author: 
'''
#拟合z=5x+6y
import numpy as np
def loadDataSet(path='.//data.txt'):
    dataMat = [];valueMat = []
    datas = open(path)
    for line in datas.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[1]),float(lineArr[2])]);
        valueMat.append(float(lineArr[0]))
    return dataMat,valueMat
#批量梯度下降
def gradAscent(dataMat,valueMat,alpha=0.01,iteration=20):
    dataMatrix = np.mat(dataMat)
    valueMatrix = np.mat(valueMat).transpose()
    m,n=np.shape(dataMatrix)
    weights = np.ones((n,1));
    weights = np.mat(weights)
    for k in range(iteration):
        preValue = dataMatrix*weights
        error = valueMatrix - preValue
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights
#随机梯度下降
def stocGradAscent(dataMatrix,valueMatrix,alpha=0.01,iteration=20):
    m,n=np.shape(dataMatrix)
    weights = np.ones(n);
    for k in range(iteration):
        for i in range(m):
            preValue = myAdd(myMulti(weights,dataMatrix[i]))
            error = valueMatrix[i] - preValue
            weights = updateWeight(weights, error, alpha, dataMatrix[i])
    return weights
#增加正则的随机梯度下降
def stocGradAscentWithReg(dataMatrix,valueMatrix,alpha=0.01,iteration=20,lam=0.2):
    m,n=np.shape(dataMatrix)
    weights = np.ones(n);
    for k in range(iteration):
        for i in range(m):
            preValue = myAdd(myMulti(weights,dataMatrix[i]))
            error = valueMatrix[i] - preValue
            tempWeights = updateWeight(weights, error, alpha, dataMatrix[i])
            weights = myMinus(tempWeights,[x/m for x in weights],lam)
            
    return weights
def myMinus(listA,listB,lam):
    return list(map(lambda x:x[0]-lam*x[1],zip(listA,listB)))
#很丑陋的4个辅助函数
def updateWeight(weights,error,alpha,listA):
    return myAddTwo(weights,list(map(lambda x:x*error*alpha,listA)))
def myMulti(listA,listB):
    return list(map(lambda x:x[0]*x[1],zip(listA,listB)))
def myAddTwo(listA,listB):
    return list(map(lambda x:x[0]+x[1],zip(listA,listB)))
def myAdd(listA):
    return reduce(lambda x,y:x+y,listA)
if __name__=='__main__':
    a,b = loadDataSet()
   # weights = gradAscent(a,b)
    hupu = stocGradAscent(a,b)
    print(hupu)
   # print(weights)
    print('main')
