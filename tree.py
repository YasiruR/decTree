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
import unittest
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

minThreshold = 0.00005
maxDepth = 4

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

# returns the entropy of a node after splitting into sub branches for numerical attributes
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

# returns split entropy of a node after splitting into sub-branches for discrete attributes
def splitEntropyDiscrete(X_d, Y):
    counters = {}
    for i in range(len(X_d)):
        if X_d[i] in counters:
            continue
        else:
            counters[X_d[i]] = 0
            
    classes = {}
    for disc_val in counters:
        classes[disc_val] = {0: 0, 1: 0}
        
    for i in range(len(X_d)):
        counters[X_d[i]] += 1
        if Y[i] == 0:
            classes[X_d[i]][0] += 1
        else:
            classes[X_d[i]][1] += 1
    
    N = len(X_d)
    E = 0
    for disc_val in counters:
        p_0 = classes[disc_val][0] / N
        p_1 = classes[disc_val][1] / N
        if p_0 == 0.0:
            E -= (counters[disc_val] / N) * (p_1 * math.log2(p_1))
        else:
            if p_1 == 0.0:
                E -= (counters[disc_val] / N) * (p_0 * math.log2(p_0))
            else:
                E -= (counters[disc_val] / N) * (p_0 * math.log2(p_0) + p_1 * math.log2(p_1))
    return E

def splitEntropyRegression(X_d, Y, s):
    above = []
    above_sum = 0
    below = []
    below_sum = 0
    for i in range(len(X_d)):
        if X_d[i] > s:
            above.append(Y[i])
            above_sum += Y[i]
        else:
            below.append(Y[i])
            below_sum += Y[i]
    
    above_mean = above_sum / len(above)
    below_mean = below_sum / len(below)
    E_above = 0
    E_below = 0
    for i in range(len(above)):
        E_above += math.pow((above[i] - above_mean), 2)
    for i in range(len(below)):
        E_below += math.pow((below[i] - below_mean), 2)
    
    return (E_above + E_below) / len(Y)

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
        
def nodeEntropyRegression(Y):
    add = 0
    N = len(Y)
    for i in range(len(Y)):
        add += Y[i]
    mean = add / N
    E = 0
    for i in range(len(Y)):
        E += math.pow((Y[i] - mean), 2)
    return E/N, mean
    
# finds the best split with the minimum entropy for the given data sets and returns both threshold value and attribute index
def split(X, Y, attr_types):
    minEnt = sys.maxsize * 2 + 1
    threshold = None
    attr_index = None
    XT = np.transpose(X)
    YT = np.transpose(Y)
    for i in range(len(XT)):
        if attr_types[i] == 'continuous':
            # finds possible split values
            splitList = findSplits(XT[i], YT[0])
            for s in range(len(splitList)):
                # calculates entropy value for each split
                E = splitEntropy(XT[i], YT[0], splitList[s])
                # returns as 0 is the minimum value 
                if E == 0.0:
                    return splitList[s], i, 'continuous'
                if E < minEnt:
                    minEnt = E
                    threshold = splitList[s]
                    attr_index = i
                    split_type = 'continuous'
        else:
            if attr_types[i] == 'discrete':    
                E = splitEntropyDiscrete(XT[i], YT[0])
                if E == 0.0:
                    return None, i, 'discrete'
                if E < minEnt:
                    minEnt = E
                    attr_index = i
                    split_type = 'discrete'
    return threshold, attr_index, split_type

def splitRegression(X, Y):
    minEnt = sys.maxsize * 2 + 1
    threshold = None
    attr_index = None
    XT = np.transpose(X)
    YT = np.transpose(Y)
    for i in range(len(XT)):
        splitList = findSplits(XT[i], YT[0])
        for s in range(len(splitList)):
            E = splitEntropyRegression(XT[i], YT[0], splitList[s])
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

def getDiscreteBranchedData(X, Y, d):
    branched_X = {}
    branched_Y = {}
    XT_d = np.transpose(X)[d]
    for i in range(len(XT_d)):
        if not XT_d[i] in branched_X:
            branched_X[XT_d[i]] = []
            branched_Y[XT_d[i]] = []
        branched_X[XT_d[i]].append(X[i])
        branched_Y[XT_d[i]].append(Y[i])
    return branched_X, branched_Y

# object resembles the tree structure recursively attached with sub-trees
class Tree:
    def __init__(self, depth, typ):
        self.threshold = None
        self.attr_index = None
        self.left_branch = None
        self.right_branch = None
        self.leaf = None
        self.discrete_branches = None
        self.depth = depth
        self.type = typ
    
    # generates the tree for given data sets X and Y
    def generate(self, X, Y, attr_types):
        if self.type == 'classification':
            E, majority = nodeEntropy(np.transpose(Y)[0])
            # returns as a leaf node if the following termination conditions are met
            if E < minThreshold or self.depth > maxDepth:
                self.leaf = majority
                return
            self.threshold, self.attr_index, split_type = split(X, Y, attr_types)
            if split_type == 'continuous':
                # returns as a lead node there are no split values found for this data set
                if self.threshold == None:
                    self.leaf = majority
                    return
                X_above, Y_above, X_below, Y_below = getBranchedData(X, Y, self.threshold, self.attr_index)
                self.left_branch = Tree(self.depth+1, self.type)
                self.right_branch = Tree(self.depth+1, self.type)
                # calls sub-trees recursively
                self.left_branch.generate(X_above, Y_above, attr_types)
                self.right_branch.generate(X_below, Y_below, attr_types)
            else:
                branched_X, branched_Y = getDiscreteBranchedData(X, Y, self.attr_index)
                self.discrete_branches = {}
                for disc_val in branched_X:
                    self.discrete_branches[disc_val] = Tree(self.depth+1, self.type)
                    self.discrete_branches[disc_val].generate(branched_X[disc_val], branched_Y[disc_val], attr_types)
        else:
            E, mean = nodeEntropyRegression(np.transpose(Y)[0])
            if E < minThreshold or self.depth > maxDepth:
                self.leaf = mean
                return
            self.threshold, self.attr_index = splitRegression(X, Y)
            if self.threshold == None:
                self.leaf = mean
                return
            X_above, Y_above, X_below, Y_below = getBranchedData(X, Y, self.threshold, self.attr_index)
            self.left_branch = Tree(self.depth+1, self.type)
            self.right_branch = Tree(self.depth+1, self.type)
            # calls sub-trees recursively
            self.left_branch.generate(X_above, Y_above, attr_types)
            self.right_branch.generate(X_below, Y_below, attr_types)
        
    def predict(self, x):
        if self.leaf != None:
            return self.leaf
        
        if self.discrete_branches == None:  
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
        else:            
            if not x[self.attr_index] in self.discrete_branches:
                return -2
            c = self.discrete_branches[x[self.attr_index]].predict(x)    
        return c
                
# util functions
# below functions carry out processing data from sample data-sets, test each data-set using the decision tree and report corresponding errors
def readDataBlobs():
    Y = []
    X = []
    with open('data/blobs.csv') as file:
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
    with open('data/flame.csv') as file:
        reader = csv.reader(file)
        line_index = 0
        for row in reader:
            if line_index > 4:
                Y.append([int(row[0][0])])
                x_list = row[0][2:].split()
                X.append([float(x_list[0]), float(x_list[1])])
            line_index += 1
    return X, Y

def readDataTicTac():
    X = []
    Y = []
    with open('data/tictac-end.csv') as file:
        reader = csv.reader(file)
        line_count = 0
        for row in reader:
            if line_count >= 7:
                Y.append([int(row[0][0])])
                x = []
                i = 2
                while True:
                    if len(x) == 9:
                        break
                    else:
                        if row[0][i] == '-':
                            i += 1
                            x.append(-(int(row[0][i])))
                        else:
                            x.append(int(row[0][i]))                            
                    i += 2
                X.append(x)
            line_count += 1
    return X, Y

def readDataTemp():
    Y = []
    X = []
    with open('data/global-temperatures.csv') as file:
        reader = csv.reader(file)
        line_index = 0
        for row in reader:
            if line_index > 6:
                X.append([int(row[0][:4])])
                Y.append([float(row[0][5:])])
            line_index += 1
    return X, Y

def readDataMpg():
    Y = []
    X = []
    with open('data/auto-mpg.csv') as file:
        reader = csv.reader(file)
        line_index = 0
        for row in reader:
            if line_index > 2:
                val_list = row[0].split()
                Y.append([float(val_list[0])])
                X.append([float(val_list[1]), float(val_list[2]), float(val_list[3]), float(val_list[4]), float(val_list[5]), float(val_list[6])])
            line_index += 1
    return X, Y

def plotPredictions(x, r, pred_list, img_name):
    sns.set_color_codes("reset")
    plt.plot(x, r, label='actual')
    plt.plot(x, pred_list, label='predicted')
    plt.legend(loc='upper left')
    plt.xlabel('year')
    plt.ylabel('global temperature')
    plt.savefig('./graphs/' + img_name + '.png')
    plt.show()
    
def testDepth(X, Y):
    depth_list = []
    r_list = []
    for i in range(19):
        global maxDepth
        maxDepth = i+1
        r_sqaure = test(X, Y, None, 'regression', None)
        depth_list.append(i+1)
        r_list.append(r_sqaure)
    plt.plot(depth_list, r_list)
    plt.xlabel('depth')
    plt.ylabel('r square')
    plt.savefig('./graphs/r_sqaure.png')
    plt.show()
    
# general test function when given lists of x and y values along with other parameters.
# attr_types - list of elements indicating whether each data type is continuous or discrete
# typ - classification/regression
# img_name - optional name for saving images
def test(X, Y, attr_types, typ, img_name):
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    sum_y = 0
    
    test_counter = 0
    for i in range(len(X)):
        if test_counter < 7:
            train_X.append(X[i])
            train_Y.append(Y[i])
        else:
            if test_counter == 9:
                test_counter = 0
            test_X.append(X[i])
            test_Y.append(Y[i])
            sum_y += Y[i][0]
        test_counter += 1        
            
    tree = Tree(0, typ)
    tree.generate(train_X, train_Y, attr_types)
    
    correct = 0
    wrong = 0
    ineligible = 0
    ss_tot = 0
    ss_res = 0
    mean_y = sum_y / len(test_Y)
    pred_list = []
    for i in range(len(test_X)):    
        c = tree.predict(test_X[i])
        if typ == 'classification':
            if c == test_Y[i][0]:
                correct += 1
            else:
                if c == -2:
                    ineligible += 1
                else:
                    wrong += 1
        else:
            pred_list.append(c)
            ss_res += math.pow((Y[i][0] - c), 2)
            ss_tot += math.pow((Y[i][0] - mean_y), 2)
                
    if typ == 'classification':
        print("accuracy: ", (correct*100.0)/(correct+wrong), "%")
        if ineligible != 0:
            print("corrected accuracy: ", (correct*100.0)/(correct+wrong-ineligible), "%")
    else:
        print("root mean square error: ", math.sqrt(ss_res/len(test_X)))   
        print("R square error: ", 1 - (ss_res / ss_tot))
        if img_name != None:
            plotPredictions(test_X, test_Y, pred_list, img_name)
        return 1 - (ss_res / ss_tot)
        
attr_types = ['continuous', 'continuous']
print('-----blobs data set (continuous)-------')
X, Y = readDataBlobs()
test(X, Y, attr_types, 'classification', None)
print()

print('-----flame data set (continuous)-------')
X, Y = readDataFlame()
test(X, Y, attr_types, 'classification', None)
print()

attr_types = ['discrete', 'discrete', 'discrete', 'discrete', 'discrete', 'discrete', 'discrete', 'discrete', 'discrete']
print('-----tic-tac data set (discrete)-------')
X, Y = readDataTicTac()
test(X, Y, attr_types, 'classification', None)
print()

print('-----global temperature data set (regression)-------')
X, Y = readDataTemp()
test(X, Y, None, 'regression', 'global_temp_pred')
print()
        
print('-----auto mpg data set (regression)-------')
X, Y = readDataMpg()
test(X, Y, None, 'regression', None)
print()
        
print('-----auto mpg test with depth (regression)-------')
testDepth(X, Y)

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
