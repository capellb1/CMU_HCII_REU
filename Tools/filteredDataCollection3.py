'''
CMU HCII REU Summer 2018
PI: Dr. Sieworek
Students:  Blake Capella & Deepak Subramanian
Date: 07/30/18

The following code is used to either train or visualize the results of a neural net. The project's goal is to analyze the
difference in performance between multi frame analysis (exercise_detector) and single frame analysis (poise_detector). Built
in to each of the files are a large number of flags used change numerous features ranging from input data
to network architecture and other hyperparameters. For more detailed information on the flags, see the code or visit 
https://github.com/capellb1/CMU_HCII_REU.git

Tool used for the final version of the model. Taking the velocity curves and applying the user determined threshold, the program
automatically determines the window of data necessary to train the model on.
'''

#Import Libraries
import math
import io

#to get rid of warning
import os
#Display libraries for Visualization2
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

from sklearn import metrics
import seaborn as sns
import glob

import statistics as stat

import numpy as np

file_names_super =[
	'Head.csv',   
	'Neck.csv',    
	'SpineShoulder.csv', 
	'SpineMid.csv',
	'SpineBase.csv',    
	'ShoulderRight.csv', 
	'ShoulderLeft.csv',  
	'HipRight.csv',
	'HipLeft.csv', 
	'ElbowRight.csv',    
	'WristRight.csv',    
	'HandRight.csv',     
	'HandTipRight.csv',  
	'ThumbRight.csv',   
	'ElbowLeft.csv',     
	'WristLeft.csv',     
	'HandLeft.csv',     
	'HandTipLeft.csv',  
	'ThumbLeft.csv',    
	'KneeRight.csv',    
	'AnkleRight.csv',   
	'FootRight.csv',     
	'KneeLeft.csv',
	'AnkleLeft.csv',     
	'FootLeft.csv']

file_names = file_names_super

dirname = os.path.realpath('.')
filename = dirname + '\\stdData\\TestNumber.txt'
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()
bodySize = 25
numSection = 1

def calcMaxEntries():
	maxEntries = 0
	timeScores = []

	for i in range(0,int(numberTests)):			
		numEntries = 0
		for line in open(dirname + "\\stdData\\test" + str(i) + "\\" + "Velocity_" + file_names_super[0] + ".csv"):
			numEntries = numEntries + 1
		if numEntries > maxEntries:
			maxEntries = numEntries	
		timeScores.append(numEntries)
	
	return maxEntries, timeScores

def extractData():
	'''
		Moves data from the text files into flattened arrays:
			[event] [HeadX1, HeadY1, HeadZ1, HeadX2....ArmX1, ArmY1, ArmZ1, ArmX2 etc]

		Parameters: None
		Returns:
			nparray shuffledlabels
			nparray shuffledData
	'''
	data =  np.empty((int(numberTests), int((bodySize)*(maxEntries*1*3))))
	print(len(data[0]))
	#enables downsampling by 50%
	labels = []

	for i in range(0, int(numberTests)):

		for j in range(0,bodySize):
			k = j*maxEntries*3
			
			for line in open(dirname + "\\stdData\\test" + str(i)+ "\\Velocity_" + file_names[j] + ".csv"):
				if k < j*maxEntries*3 + maxEntries*3:
					row = line.split(',')
					for l in range(0,3):
						data[i][k] = row[l]
						k = k +1
			
		#seperate the label from the name and event number stored within the label.csv file(s)
		for line in open(dirname + "\\stdData\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			labels.append(str(temporaryLabel[0]))
	return data, labels

bodySize = 25
maxEntries, timeScores = calcMaxEntries()
velocityData, labels = extractData()

for i in range(0, int(numberTests)):
	offset = 0
	newDir2 = dirname + "\\DataWindow\\test" + str(i)
	if not (os.path.exists(newDir2)):
		os.makedirs(newDir2)

	for line in open(dirname + "\\stdData\\test" + str(i)+ "\\label.csv"):
		temporaryLabel = line.split()
		temporaryLabel = temporaryLabel[0]
	
	
	resultsFileL = open(dirname + "\\DataWindow\\test" + str(i) +"\\label.csv", "w")
	resultsFileL.write(temporaryLabel)
	#A
	velocityAboveThreshold = np.zeros((25, timeScores[i]))

	for j in range(0,25):
		#print("new body")
		l = 0
		for k in range(offset, offset + maxEntries*3):
			if k%(maxEntries*3) < timeScores[i]*3:
				if (k%3 == 0):
					dataPointScore = math.fabs(velocityData[i][k]) + math.fabs(velocityData[i][k+1]) + math.fabs(velocityData[i][k+2])
					if (dataPointScore > .75):
						velocityAboveThreshold[j][l] = 1
					else:
						velocityAboveThreshold[j][l] = 0
					l = l + 1

		offset = offset + maxEntries*3

	maxConsecutive = np.zeros((25))
	consecutiveOnes = np.zeros((25))
	#print("Test", i)
	for j in range (0,25):
		numberOnes = 0
		for k in range (0, timeScores[i]):
			if (velocityAboveThreshold[j][k] == 1):
				numberOnes = numberOnes + 1
		#print("Percent 1s", (numberOnes/timeScores[i])*100, '%')

	for j in range(0,25):
		for k in range (0, timeScores[i]):
			if (velocityAboveThreshold[j][k] == 1):
				consecutiveOnes[j] = consecutiveOnes[j] + 1
			else:
				if (maxConsecutive[j] < consecutiveOnes[j]):
					maxConsecutive[j] = consecutiveOnes[j]
				consecutiveOnes[j] = 0
	
	timeStamps = np.zeros((50))
	
	#print("FINDING RANGES PER BODYPART")
	for j in range (0, 25):
		#print("NEW BODYPART")
		firstLoopFound = 0
		secondLoopFound = 0
		currentConsecutive = 0
		done = 0
		for k in range (0, timeScores[i]):
			if not done:
				if(firstLoopFound == 0 and velocityAboveThreshold[j][k] == 0 and currentConsecutive >= 4):
					#if (i == 0):
					#	print("first loop bodypart", j,"timestamp", k)
					timeStamps[2*j] = k 
					firstLoopFound = 1
					currentConsecutive = 0

				elif(firstLoopFound == 0 and velocityAboveThreshold[j][k] == 0 and currentConsecutive != 4):
					currentConsecutive = 0

				elif(firstLoopFound == 0 and velocityAboveThreshold[j][k] == 1 ):			
					currentConsecutive = currentConsecutive + 1

				elif (firstLoopFound == 1 and velocityAboveThreshold[j][k] == 1 and currentConsecutive != 3):
					currentConsecutive = currentConsecutive + 1

				elif(firstLoopFound == 1 and velocityAboveThreshold[j][k] == 1 and currentConsecutive == 3):
					#if (i == 0):
					#	print("second loop bodypart", j,"timestamp", k)
					timeStamps[2*j + 1] = k - 4
					firstLoopFound = 0
					done = 1
					currentConsecutive = 0
		firstLoopFound = 0
		currentConsecutive = 0

	print("FINDING FINAL RANGE OF Test ", i)
	range1 = 10000000000
	range2 = -10000000000
	for j in range (0, 25):
		#print("first loop", timeStamps[2*j], "second loop", timeStamps[2*j + 1])
		if (timeStamps[2*j]!=0 and timeStamps[2*j+1]!=0 and timeStamps[2*j] < range1):
			range1 = timeStamps[2*j]
		if (timeStamps[2*j]!=0 and timeStamps[2*j+1]!=0 and timeStamps[2*j+1] > range2):
			range2 = timeStamps[2*j+1]

	numberBodyPartsInRange = 0
	for j in range(0,25):
		if (timeStamps[2*j] < timeStamps[2*j + 1] and timeStamps[2*j] >= range1 and timeStamps[2*j+1] <= range2):
			numberBodyPartsInRange = numberBodyPartsInRange + 1

	print("ranges ", range1, range2)
	print("maxTime ", timeScores[i])
	print("# of Body Parts in Range ", numberBodyPartsInRange)


	k = 0
	for j in range(0,bodySize):
		resultsFileP = open(dirname + "\\DataWindow\\test" + str(i) + "\\Position_" + file_names[j], "a+")
		m = 0
		sample = True
		for line in open(dirname + "\\stdData\\test" + str(i)+ "\\Position_" + file_names[j] + ".csv"):
			if m >= range1 and m <= range2:
				if sample:
					row = line.split(',')
					coords = []
					for l in range(0,3):
						coords.append(row[l])
						k = k +1
						sample = False
					coords = coords[0] + "," + coords[1] + "," + coords[2] 
					resultsFileP.write(coords) 
				else:
					sample = True
			m = m + 1
		resultsFileV = open(dirname + "\\DataWindow\\test" + str(i) + "\\Velocity_" + file_names[j], "a+")
		m = 0
		sample = True
		for line in open(dirname + "\\stdData\\test" + str(i)+ "\\Velocity_" + file_names[j] + ".csv"):
			if m >= range1 and m <= range2:
				if sample:
					row = line.split(',')
					coords = []
					for l in range(0,3):
						coords.append(row[l])
						k = k +1
						sample = False
					coords = coords[0] + "," + coords[1] + "," + coords[2] 
					resultsFileV.write(coords) 
				else:
					sample = True
			m = m + 1
		'''
		resultsFileT = open(dirname + "\\DataWindow\\test" + str(i) + "\\Task_" + file_names[j], "a+")
		m = 0
		sample = True
		for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\Task_" + file_names[j]):
			if m >= range1 and m <= range2:
				if sample:
					coords = []
					row = line.split(',')
					for l in range(0,3):
						coords.append(row[l])
						k = k +1
						sample = False
					coords = coords[0] + "," + coords[1] + "," + coords[2] 
					resultsFileT.write(coords) 
				else:
					sample = True
			m = m + 1
		'''
print ("done")

