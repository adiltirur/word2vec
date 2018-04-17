import csv
import random
import math
import operator
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


# KNN algorithm for prediction

#training dataset building
def loadDataset(filename,split,  trainingSet=[] ):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(1):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            trainingSet.append(dataset[x])


#testing dataset building
def loadDataset1(filename1,split, testSet=[]):
	with open(filename1, 'rb') as csvfile1:
	    lines1 = csv.reader(csvfile1)
	    dataset1 = list(lines1)
	    for x in range(len(dataset1)-1):
	        for y in range(1):
	            dataset1[x][y] = float(dataset1[x][y])
	        if random.random() < split:
	            testSet.append(dataset1[x])
	        else:
	            testSet.append(dataset1[x])


# calculating the euclidean distance between each inputs so then we can use KNN
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)

# Finding the neighbors using the euclidean distance so this gives us a point in the space
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
#get the prediction
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
#finding the accuracy (I have disabled it for the test data and used it only with the dev folder)
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 1
	loadDataset('/home/adil/Desktop/train.csv',split, trainingSet)
	loadDataset1('/home/adil/Desktop/test.csv',split,  testSet)
	print(trainingSet[0][-1])
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	actual=[]
	k = 3

	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		actual.append(testSet[x][1])
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		print (repr(testSet[x][-1])+'\t'+repr(result))
		#saving the result into the specified format
		f=open('/home/adil/Desktop/word2vec/test_answers.tsv','ab')
		f.write((testSet[x][-1])+'\t'+(result)+'\n')

	cm=(confusion_matrix(actual, predictions))
	print(cm)
	#plotting confusion matrix
	df_cm = pd.DataFrame(cm, range(34),
                  range(34))
	plt.figure(figsize = (34,34))
	sn.set(font_scale=1)#for label size
	sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
	plt.savefig('/home/adil/Desktop/confusion.png')



main()
