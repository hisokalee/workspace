#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class MyLinearRegressor():

    def __init__(self, kappa=0.01, lamb=0, max_iter=200, opt='batch'):
        self._kappa = kappa
        self._lamb = lamb
        self._opt = opt
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        error = []
        if self._opt == 'sgd':
            error = self.__stochastic_gradient_descent(X, y)
        elif self._opt == 'batch':
            error = self.__batch_gradient_descent(X, y)
        elif self._opt == 'isgd':
            error = self.__improved_stochastic(X, y)
        else:
            print 'unknow opt'
        return error

    def predict(self, X):
        pass

    def __batch_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        self._w = np.ones(X.shape[1])
        ##############################
        #
        #  put your code here
        # �Ѿ����һ���֣���֪���Բ���
        # ��ʵ��Ŀ���ѵ�
        w = self._w
        while niter<self._max_iter:
            error.append(self.__total_error(X,y,w))
            #gradient= -np.sum(np.dot(X,(y-np.dot(X,w))))/len(y)
            #w=w-self._kappa*gradient
            w = w + alpha*X*np.mat(error).transpose()
            niter+=1
        #
        ##############################
        return error

    def __stochastic_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        self._w = np.ones(X.shape[1])
        ##############################
        #
        #  put your code here
        #
        ##############################
        while niter <self._max_iter:
            errorSum = 0
            for k in range(N):
                errorSum += y[k]-X[k]*self._w
                w = w + error*alpha*X[k]
            error.append(self.__total_error(X,y,w))
        return error

    def __improved_stochastic(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        self._w = np.ones(X.shape[1])
        G = np.zeros((X.shape[1], X.shape[1]))
        ##############################
        #
        #  put your code here
        #
        ##############################
        return error

    def __total_error(self, X, y, w):
        tl = 0.5 * np.sum((np.dot(X, w) - y)**2)/len(y)
        return tl

    # add a column of 1s to X
    def __feature_prepare(self, X_):
        M, N = X_.shape
        X = np.ones((M, N+1))
        X[:, 1:] = X_
        return X

    # rescale features to mean=0 and std=1
    def __feature_rescale(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        return (X - self._mu)/self._sigma


if __name__ == '__main__':
    #from sklearn.datasets import load_boston

    #data = load_boston()
    #X, y = data['data'], data['target']
    X=np.mat([[]])
    mylinreg = MyLinearRegressor()
    mylinreg.fit(X, y)
