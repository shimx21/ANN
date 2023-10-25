from __future__ import division
import numpy as np


class MSELoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        self.delta = input - target
        return np.mean(np.sum(self.delta ** 2, axis=1))
        # TODO END

    def backward(self, input, target):
		# TODO START
        '''Your codes here'''
        n = input.shape[0]
        return 2 * self.delta / n
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        expInp = np.exp(input)
        self.softMax = expInp / np.sum(expInp, axis = 1, keepdims=True)
        loss = -np.mean(np.sum(target * np.log(self.softMax + 0.001), axis=1))
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        return (self.softMax - target)/target.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        '''Your codes here'''
        xtn = np.sum(input * target, axis=1, keepdims=True)
        self.hinge = self.margin  - xtn + input
        self.hinge[((target == 1) | (self.hinge < 0))] = 0
        return np.mean(np.sum(self.hinge, axis=1))
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        self.hinge[self.hinge > 0] = 1
        self.hinge[target == 1] = -np.sum(self.hinge, axis=1)
        return self.hinge/target.shape[0]
        # TODO END


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=0.5):
        self.name = name
        if alpha is None:
            self.alpha = np.array([0.1 for _ in range(10)])
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        expInp = np.exp(input)
        # Store for backward()
        self.h = expInp / np.sum(expInp, axis = 1, keepdims=True)
        logh = np.log(self.h)
        self.gradCoe = np.sum(np.where(target == 1, self.alpha * 
                                       np.power(1 - self.h, self.gamma - 1) * (
                                       - logh * self.gamma * self.h + 
                                       1 - self.h
                                       ), 0), axis=1, keepdims=True)
        
        # Calculte loss
        # coe1 = self.alpha * target + (1 - self.alpha) * (1 - target)
        # coe2 = np.power(1 - self.h, self.gamma)
        # return np.mean(np.sum(coe1 * coe2 * self.base * target, axis=1))
        # Simplized for one-hot label:
        return  - np.mean(np.sum(np.where(target == 1, 
                                  logh * np.power(1 - self.h, self.gamma), 
                                  0) * self.alpha, axis=1))
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        return self.gradCoe * (self.h - target) / target.shape[0]
        # TODO END
