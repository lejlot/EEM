"""
EXTREME ENTROPY MACHINES

Implementation assumes that there are two integer coded labels.
Note that this model is suited for maximizing balanced accuracy
(or GMean, MCC) and should not be used if your task is to maximize
accuracy (or other imbalanced metric).
"""

import numpy as np
from scipy import linalg as la
from sklearn.covariance import LedoitWolf

def sigmoid(X, W, b):
    """ Basic sigmoid activation function """
    return 1./(1. + np.exp(X.dot(W.T) - b))

def relu(X, W, b):
    """ Basic rectified linear unit """
    return np.maximum(0, X.dot(W.T) - b)

def tanimoto(X, W, b=None):
    """ Tanimoto similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return XW / (XX+WW-XW)

def sorensen(X, W, b=None):
    """ Sorensen similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return 2 * XW / (XX+WW)

class EEM(object):
    """ 
    Extreme Entropy Machine

    as presented in 
    "Extreme Entropy Machines: Robust information theoretic classification",
    WM Czarnecki and J Tabor,
    Pattern Analysis and Applications (2015)
    DOI: 10.1007/s10044-015-0497-8    
    http://link.springer.com/article/10.1007/s10044-015-0497-8
    """

    def __init__(self, h='sqrt', f=tanimoto, C=10000, random_state=None, from_data=True):
        """
        h - number of hidden units, can be
         i) integer, giving the exact number of units
         ii) float, denoting fraction of training set to use (requires: from_data=True)
         iii) string, one of "sqrt", "log", with analogous meaning as the above
        f - activation function (projection)
        C - inverse of covariance estimation smoothing or None for a minimum possible
        from_data - whether to select hidden units from training set (prefered)
        random_state - seed for random number genrator
        """

        self.h = h
        self.C = C
        self.f = f
        self.rs = random_state
        self.fd = from_data
        self._maps = {'sqrt': np.sqrt, 'log': np.log}
        if isinstance(self.h, float):
            if not self.fd:
                raise Exception('Using float as a number of hidden units requires learning from data')
        if isinstance(self.h, str):
            if not self.fd:
                raise Exception('Using string as a number of hidden units requires learning from data')
            if self.h not in self._maps:
                raise Exception(self.h + ' is not supported as a number of hidden units')


    def _pdf(self, x, l):
        """ Returns pdf og l'th class """
        return 1. / np.sqrt(2 * np.pi * self.sigma[l]) * np.exp( -(x - self.m[l]) ** 2 / (2 * self.sigma[l]))

    def _hidden_init(self, X, y):
        """ Initializes hidden layer """
        np.random.seed(self.rs)
        if self.fd:
            if isinstance(self.h, float):
                self.current_h = self.h * X.shape[0]
            if isinstance(self.h, str):
                self.current_h = self._maps[self.h](X.shape[0])
            self.current_h = max(1, min(self.current_h, X.shape[0]))
            W = X[np.random.choice(range(X.shape[0]), size=self.current_h, replace=False)]
        else:      
            self.current_h = self.h  
            W = csr_matrix(np.random.rand(self.current_h, X.shape[1]))
        b = np.random.normal(size=self.current_h)
        return W, b

    def fit(self, X, y):
        """ Trains the model """

        self.W, self.b = self._hidden_init(X, y)        
        H = self.f(X, self.W, self.b)
        self.labels = np.array([np.min(y), np.max(y)])
        self.m = [0, 0]
        self.sigma = [0, 0]

        for l in range(2):
            data = H[y==self.labels[l]]
            self.m[l] = np.mean(data, axis=0)
            self.sigma[l] = LedoitWolf().fit(data).covariance_
            if self.C is not None:
                self.sigma[l] += np.eye(self.current_h) / (2.0*self.C)

        self.beta = la.pinv(self.sigma[0] + self.sigma[1]).dot(self.m[1] - self.m[0])
        for l in range(2):
            self.m[l] = self.beta.T.dot(self.m[l])
            self.sigma[l] = self.beta.T.dot(self.sigma[l]).dot(self.beta)

    def predict(self, X):
        """ Labels given set of samples """

        p = self.f(X, self.W, self.b).dot(self.beta)
        result = np.argmax([self._pdf(p, l) for l in range(2)], axis=0)        
        return self.labels[result]

    def predict_proba(self, X):
        """ Returns probability estimates """

        p = self.f(X, self.W, self.b).dot(self.beta)
        result = np.array([self._pdf(p, l) for l in range(2)]).T
        return result / np.sum(result, axis=1).reshape(-1, 1)

if __name__ == '__main__':
    from time import time
    
    data = []
    T = 100000
    for e in range(1, 9):
        N = int(10 ** (e/2.))

        X = np.random.randn(10 * N).reshape(N, -1) / 3
        Xt = np.random.randn(10 * T).reshape(T, -1) / 3 

        noise = (np.random.rand(10 * N) * 2 - 1).reshape(N, -1)
        noiset = (np.random.rand(10 * T) * 2 - 1).reshape(T, -1)

        print N
        for clf in [EEM(f=relu, h='sqrt', random_state=0)]: 
            print clf.__class__.__name__
            s = time()
            trainX = np.vstack((X, noise))
            trainy = np.array( [0] * N + [1] * N )
            clf.fit(trainX, trainy)
            ttime = time() - s
            print 'train', ttime
            s = time()
            score = 0.5 * np.mean(clf.predict(Xt) == 0) + 0.5 * np.mean(clf.predict(noise) != 0)
            print 'BAC', score 
            tetime = time() - s
            print 'test', tetime
            print
            #print clf.predict_proba(Xt)
