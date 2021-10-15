#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 08:22:40 2021

@author: yasi
"""

import sys
import numpy as np
import math
import csv
from matplotlib import pyplot as plt
import unittest

minThreshold = 0.00005
maxDepth = 7

# takes params as ith row of X transpose (i.e. set of data corresponding to a certain dimension)
def findSplits(XT_i, Y):
    # adding both value sets to a dictionary
    dic = {}
    for i in range(len(XT_i)):
        # if the same value is already added to dictionary check if y is different and if so proceed with adding
        if XT_i[i] in dic:
            if dic[XT_i[i]] != Y[i]:
                dic[XT_i[i] + 0.00001] = Y[i]
        else:
            dic[XT_i[i]] = Y[i]
    
    # sorting X
    XT_i.sort()
    splitList = []
    for k in range(len(XT_i)):
        # checks if it is not the last element
        if k < len(XT_i) - 1:
            # adjacent x values with only opposite y values are considered
            if dic[XT_i[k]] != dic[XT_i[k + 1]]:
                # takes the splitting value as average of 2 values
                splitList.append((XT_i[k] + XT_i[k+1]) / 2)
    return splitList

# returns the entropy of a node after splitting into sub branches
def splitEntropy(X_d, Y, s):
    above_0 = 0
    above_1 = 0
    below_0 = 0
    below_1 = 0
    for i in range(len(X_d)):
        if (X_d[i] > s):
            if Y[i] == 0:
                above_0 += 1
            else:
                above_1 += 1
        else:
            if Y[i] == 0:
                below_0 += 1
            else:
                below_1 += 1
    
    p_above_0 = above_0/(above_0 + above_1)
    if p_above_0 == 0:
        E_above = -math.log2(1 - p_above_0)
    else:
        if p_above_0 == 1:
            E_above = -math.log2(p_above_0)
        else:
            E_above = -(p_above_0 * math.log2(p_above_0) + (1 - p_above_0) * math.log2(1 - p_above_0))
    
    p_below_0 = below_0/(below_0 + below_1)
    if p_below_0 == 0:
        E_below = -math.log2(1 - p_below_0)
    else:
        if p_below_0 == 1:
            E_below = -math.log2(p_below_0)
        else:
            E_below = -(p_below_0 * math.log2(p_below_0) + (1 - p_below_0) * math.log2(1 - p_below_0))
    
    N = above_0 + above_1 + below_0 + below_1
    return ((above_0 + above_1) / N) * E_above + ((below_0 + below_1) / N) * E_below

# calculates the entropy of a node along with the majority class
def nodeEntropy(Y):
    c0 = 0
    c1 = 0
    for i in range(len(Y)):
        if Y[i] == 1:
            c1 += 1
        else:
            c0 += 1
    
    p0 = c0 / (c0 + c1)
    if c0 > c1:
        majority = 0
    else:
        majority = 1
    
    if p0 == 0:
        return -math.log2(1 - p0), majority
    else:
        if p0 == 1:
            return -math.log2(p0), majority
        else:
            return -(p0 * math.log2(p0) + (1 - p0) * math.log2(1 - p0)), majority
    
# finds the best split with the minimum entropy for the given data sets and returns both threshold value and attribute index
def split(X, Y):
    minEnt = sys.maxsize * 2 + 1
    threshold = None
    attr_index = None
    XT = np.transpose(X)
    YT = np.transpose(Y)
    for i in range(len(XT)):
        # finds possible split values
        splitList = findSplits(XT[i], YT[0])
        for s in range(len(splitList)):
            # calculates entropy value for each split
            E = splitEntropy(XT[i], YT[0], splitList[s])
            # returns as 0 is the minimum value 
            if E == 0.0:
                return splitList[s], i
            
            if E < minEnt:
                minEnt = E
                threshold = splitList[s]
                attr_index = i
    return threshold, attr_index

# branches the data set using split value and the relevant dimension provided
def getBranchedData(X, Y, s, d):
    X_above = []
    Y_above = []
    X_below = []
    Y_below = []
    XT_d = np.transpose(X)[d]
    for i in range(len(XT_d)):
        if XT_d[i] > s:
            X_above.append(X[i])
            Y_above.append(Y[i])
        else:
            X_below.append(X[i])
            Y_below.append(Y[i])
    return X_above, Y_above, X_below, Y_below
            
# object resembles the tree structure recursively attached with sub-trees
class Tree:
    def __init__(self, depth):
        self.threshold = None
        self.attr_index = None
        self.left_branch = None
        self.right_branch = None
        self.leaf = None
        self.depth = depth
    
    # generates the tree for given data sets X and Y
    def generate(self, X, Y):
        E, majority = nodeEntropy(np.transpose(Y)[0])
        # returns as a leaf node if the following termination conditions are met
        if E < minThreshold or self.depth > maxDepth:
            self.leaf = majority
            return
        
        self.threshold, self.attr_index = split(X, Y)
        # returns as a lead node there are no split values found for this data set
        if self.threshold == None:
            self.leaf = majority
            return
        
        X_above, Y_above, X_below, Y_below = getBranchedData(X, Y, self.threshold, self.attr_index)
        
        self.left_branch = Tree(self.depth+1)
        self.right_branch = Tree(self.depth+1)
        
        # calls sub-trees recursively
        self.left_branch.generate(X_above, Y_above)
        self.right_branch.generate(X_below, Y_below)
        
    def predict(self, x):
        if self.leaf != None:
            return self.leaf
        if self.attr_index == 0:
            if x[0] > self.threshold:
                c = self.left_branch.predict(x)
            else:
                c = self.right_branch.predict(x)
        else:
            if x[1] > self.threshold:
                c = self.left_branch.predict(x)
            else:
                c = self.right_branch.predict(x)
        return c
                
def readDataBlobs():
    Y = []
    X = []
    with open('blobs.csv') as file:
        reader = csv.reader(file)
        line_index = 0
        for row in reader:
            if line_index > 1:
                Y.append([int(row[0][0])])
                x_list = row[0][2:].split()
                X.append([float(x_list[0]), float(x_list[1])])
            line_index += 1
    return X, Y

def readDataFlame():
    Y = []
    X = []
    with open('flame.csv') as file:
        reader = csv.reader(file)
        line_index = 0
        for row in reader:
            if line_index > 4:
                Y.append([int(row[0][0])])
                x_list = row[0][2:].split()
                X.append([float(x_list[0]), float(x_list[1])])
            line_index += 1
    return X, Y

def crossValidate(X, Y):
    correct = 0
    wrong = 0
    for i in range(len(X)):
        tmp_X = np.delete(X, i, axis=0)
        tmp_Y = np.delete(Y, i, axis=0)
        
        tree = Tree(0)
        tree.generate(tmp_X, tmp_Y)
        c = tree.predict(X[i])
        
        if Y[i][0] == c:
            correct += 1
        else:
            wrong += 1
    print("accuracy: ", (correct*100.0)/(correct+wrong), "%")

def plotData(X, Y):
    x_0 = []
    x_1 = []
    y_0 = []
    y_1 = []
    for i in range(len(X)):
        if Y[i][0] == 0:
            x_0.append(X[i][0])
            y_0.append(X[i][1])
        else:
            x_1.append(X[i][0])
            y_1.append(X[i][1])
    
    plt.scatter(x_0, y_0, color='green')
    plt.scatter(x_1, y_1, color='brown')
    
    plt.axvline(x = 0.5, color = 'r', linestyle = '--')
    plt.axhline(y = 0.4314, color = 'b', linestyle = '-')
    plt.show()

X, Y = readDataBlobs()
crossValidate(X, Y)

X, Y = readDataFlame()
crossValidate(X, Y)

#plotData(X, Y)

# unit tests #
class Test(unittest.TestCase):
    def test_findSplits(self):
        self.assertEqual(findSplits([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 1, 0, 1, 1]), [1.5, 2.5, 3.5])
        
    def test_splitEntropy(self):
        self.assertAlmostEqual(splitEntropy([4.5, 4.4, 4.3, 5.8, 3.2, 3.0, 2.5, 3.9, 1.0, 1.5, 2.1, 3.0, 3.1, 4.0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], 4.2), 0.91520778516)
        self.assertEqual(splitEntropy([-0.6674, -0.6605, -0.6067], [0, 0, 1], -0.6335999999999999), 0.0)

    def test_nodeEntropy(self):
        self.assertAlmostEqual(nodeEntropy([0, 0, 1, 1, 0, 1, 0, 1, 1]), (0.9910760598382222, 1))
        
    def test_split(self):
        self.assertEqual(split([[2.0, 3.0], [2.0, 5.0], [2.0, 4.0]], [[0], [1], [1]]), (3.5, 1))
        self.assertEqual(split([[2.0, 3.0], [4.0, 5.0], [4.0, 4.0], [3.0, 3.0]], [[0], [1], [1], [1]]), (2.5, 0))                
        
    def test_getBranchedData(self):
        self.assertEqual(getBranchedData([[11, 13], [8, 7], [1, 11], [11, 1]], [[0], [1], [1], [0]], 10, 1), ([[11, 13], [1, 11]], [[0], [1]], [[8, 7], [11, 1]], [[1], [0]]))
        self.assertEqual(getBranchedData([[11, 13], [8, 7], [1, 11], [11, 1]], [[0], [1], [1], [0]], 9, 0), ([[11, 13], [11, 1]], [[0], [0]], [[8, 7], [1, 11]], [[1], [1]]))
