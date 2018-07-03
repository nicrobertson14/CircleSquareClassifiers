import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import math
import copy
import time

tree = None

class Node():
    def __init__(self,att,split):
        self.attribute = att
        self.split = split
        self.left = self.right = None

def frange(lower,upper,step,div):
    return [x / div for x in range(lower, upper, step)]

def mode(image_tuples):
    count = 0
    for tuple in image_tuples:
        count += tuple[1] 
        if(count > len(image_tuples)/2):
            return 1
    return 0

# returns the entropy of a set of images
def Entropy(image_tuples):
    c1 = 0
    size = len(image_tuples)
    for tuple in image_tuples:
        c1 += tuple[1]
    c2 = size - c1
    
    # check for perfect split
    if(c1 * c2 == 0):
        return 0
    return -1 * (c1/size) * math.log2(c1/size) - (c2/size) * math.log2(c2/size)

# returns the information gain after splitting a set of images
def InformationGain(before,left,right):
    size = len(before)
    ent_before = Entropy(before)
    ent_left = Entropy(left)
    ent_right=Entropy(right)
    ent_after = (len(left)/size) * ent_left + (len(right)/size)*ent_right
    
    return ent_before - ent_after

# Iterates through possible thresholds/attributes to find highest information gain
def ChooseAttribute(attributes,image_tuples):
    thresholds = frange(1,100,2,100.0)
    best_ig = -1
    less = []
    greater = []
    
    for attribute in attributes:
        for threshold in thresholds:
            for tuple in image_tuples:
                if(tuple[0][attribute] >= threshold):
                    greater.append(tuple)
                else:
                    less.append(tuple)
            inf_gain = InformationGain(image_tuples,less,greater)
            if(inf_gain > best_ig):
                best_ig = inf_gain
                best_att = attributes.index(attribute)
                best_left = copy.copy(less)
                best_right = copy.copy(greater)
                best_thresh = threshold
            less.clear()
            greater.clear()
            
    return [best_att,best_thresh,best_left,best_right]

# checks if all labels in a list are the same
def CheckLabels(image_tuples):
    for i in range(len(image_tuples)-1):
       if(image_tuples[i][1] != image_tuples[i+1][1]):
           return False
       
    return True

# Recursively builds a decision tree, split by attribute with greatest information gain
def DTL(attributes,image_tuples,default,prune_factor):
    if(len(image_tuples) == 0):
        return default
    elif(len(attributes) == 0):
        return mode(image_tuples)
    elif(Entropy(image_tuples) < prune_factor):
        return mode(image_tuples)
    elif(CheckLabels(image_tuples) == True):
        return image_tuples[0][1]
    else:
        best_attr,split,left,right = ChooseAttribute(attributes,image_tuples)
        tree = Node(best_attr,split)
        
        #remove best attribute
        del attributes[best_attr]
        tree.left = DTL(attributes,left,mode(image_tuples),prune_factor)
        tree.right= DTL(attributes,right,mode(image_tuples),prune_factor)
        
        return tree

# Recursively searches tree returning predicted label
def SearchDT(tree,image_tuple):
    if(tree == 0 or tree == 1):
        return tree
    elif(image_tuple[0][tree.attribute] < tree.split):
        return SearchDT(tree.left,image_tuple)
    else:
        return SearchDT(tree.right,image_tuple)
    
def TestClassifier(image_tuples):
    global tree
    correct_count = 0
    for tuple in image_tuples:
        best = SearchDT(tree,tuple)
        if(best == tuple[1]):
            correct_count += 1

    return correct_count/len(image_tuples)
    
### main ###

dataset = scipy.io.loadmat('dataset.mat')
trainImages = dataset['train_image'].reshape(200,576)
trainLabels = dataset['train_label']
testImages = dataset['test_image'].reshape(200,576)
testLabels = dataset['test_label']

# Re-arrange data into train/test lists of tuples
train_tuples = []
test_tuples = []
for i in range(0,200,1):
    train_tuples.append([trainImages[i,:],trainLabels[i][0]])
    test_tuples.append([testImages[i,:],testLabels[i][0]])
    
tree = DTL(list(range(0,576)),train_tuples,None,0.0)
print(TestClassifier(test_tuples))


