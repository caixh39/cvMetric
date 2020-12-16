# -*- coding: utf-8 -*-

class AverageMeter(object):
    """
    Average meter

    usage: 
    ----------------------------
    >>> import random 
    >>> metric = ["acc", "error"]
    >>> test = AverageMeter(*metric)
    >>> N = 10
    >>> for _ in range(N):
            tmp = {"acc": random.random(), "error": random.random()}
            test.add(**tmp)
    >>> print(test.getAvg("acc"))
    >>> print(test.getAvg(*metric))
    """
    def __init__(self, *args):
        self.args = args
        self.reset()

    def reset(self):
        self.sum = {i : 0.0 for i in self.args}
        self.cnt = {i : 0 for i in self.args}

    def add(self, **kwargs):  
        for key, item in kwargs.items():
            self.sum[key] += item
            self.cnt[key] += 1

    def getAvg(self, *args):
        avg = {i : None for i in args}
        for key in args:
            avg[key] = self.sum[key]/self.cnt[key]
        return avg


class Accumulator(object):
    '''
    Sum a list of numbers over time
    '''
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a+b for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0] * len(self.data)
        
    def __getitem__(self, i):
        return self.data[i]
