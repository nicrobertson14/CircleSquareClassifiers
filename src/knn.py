import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import spatial
import math
import sys
from operator import itemgetter
import time

#global variables
tree = None
neighbours = ()

class Node():
    def __init__(self,value):
        self.data = value
        self.left = self.parent = self.right = None

# maintains a sorted list of k nearest neighbours
def addNeighbour(nodetuple):
    global neighbours
    if(nodetuple[0] >= neighbours[-1][0]):
        return
    for neighbour in neighbours:
        if(nodetuple[0] == neighbour[0]):
            return
    neighbours[-1] = nodetuple
    neighbours = sorted(neighbours, key=itemgetter(0))

# returns the euclidean distance between two images
def Euclidean(image1,image2):
    sqDiff = np.square(image1-image2)
    sum = np.sum(sqDiff)
    return np.sqrt(sum)
    
# returns the lower bound distance between a node and query point
def distanceLB(node,query):
    return abs(node.image[node.d] - query[node.d])

# returns the median value of a list
def getMedian(list):
    copy = list.copy()
    copy.sort()
    index = int(len(copy)/2)
    return [copy[index]]

# checks if input node is a leaf
def isLeaf(node):
    return node.left == None and node.right == None
    
# recursively builds k-d tree, sorting and splitting data at median value
def makeKdTree(images,labels,feature):
    size = len(images)
    if(size == 0):
        return

    idx = np.argsort(images[:,feature])
    sorted_images = images[idx,:]
    sorted_labels = labels[idx,:]
    median = getMedian(sorted_images[:,feature])
    median_index = int(len(sorted_images[:,feature])/2) 

    root = Node(None)
    root.data = median
    root.d = feature
    root.image = sorted_images[median_index,:]

    if(size == 1):
        root.leaf = "LEAF"
        root.label = sorted_labels[median_index,:][0]
        return root
    
    root.left = makeKdTree(sorted_images[0:median_index,:],sorted_labels[0:median_index,:],feature+1)
    root.right = makeKdTree(sorted_images[median_index:size,:],sorted_labels[median_index:size,:],feature+1)

    return root

# recursively searches for closest image to query image
def searchKdTree(root,input,best):
    global neighbours
    if(root == None):
        return
    if(isLeaf(root)):
        addNeighbour((Euclidean(root.image,input),root.label))
        if(best == -1):
            return
    
    if(input[root.d] < root.data):
        child_near = root.left
        child_far = root.right
    else:
        child_near = root.right
        child_far = root.left
    
    searchKdTree(child_near,input,best)
    last = len(neighbours) - 1
    if(distanceLB(root,input) < neighbours[last][0] or neighbours[last][0] == math.inf):
        searchKdTree(child_far,input,best)

# predicts image label based on majority vote from k nearest neighbours
def knnVote(neighbours):
    size = len(neighbours)
    count = 0
    for neighbour in neighbours:
        if(neighbour[1] == 1):
            count += 1

    return count > int(size/2)

# searches k-d tree for knn of each test image and prints prediction accuracy
def testKNNClassifier(images,labels,k):
    global tree, neighbours
    correct_count = 0
    for i in range(labels.size):
        neighbours = [(math.inf,math.inf)] * k
        searchKdTree(tree,images[i,:],-1)

        if(knnVote(neighbours) == labels[i,:][0]):
            correct_count += 1

        neighbours.clear()
    
    return correct_count/len(images)


### main ###

dataset = scipy.io.loadmat('./img-data/dataset.mat')
trainImages = dataset['train_image'].reshape(200,576)
trainLabels = dataset['train_label']
testImages = dataset['test_image'].reshape(200,576)
testLabels = dataset['test_label']
k = int(sys.argv[1])

tree = makeKdTree(trainImages,trainLabels,0)
print(testClassifier(testImages,testLabels,k))
