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

Current method of standardizing data per individual. Only works on position data, see other file in tools folder for the velocity
variant
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
filename = dirname + '\\selectedData\\TestNumber.txt'
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()
bodySize = 25
numSection = 1

def calcMaxEntries():
	maxEntries = 0
	timeScores = []

	for i in range(0,int(numberTests)):			
		numEntries = 0
		for line in open(dirname + "\\selectedData\\test" + str(i) + "\\" + "Position_" + file_names_super[0]):
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
	data =  np.empty((int(numberTests), int((bodySize)*(math.floor(maxEntries//2)*1*3))))
	#enables downsampling by 50%
	labels = []

	for i in range(0, int(numberTests)):

		for j in range(0,bodySize):
			sample = True

			k = j*math.floor(maxEntries//2)*3
			
			for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\Position_" + file_names[j]):
				if k < j*math.floor(maxEntries//2)*3 + math.floor(maxEntries//2)*3:
					if sample:
						row = line.split(',')
						for l in range(0,3):
							data[i][k] = row[l]
							k = k +1
							sample = False

					else:
						sample = True
			
		#seperate the label from the name and event number stored within the label.csv file(s)
		for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			labels.append(str(temporaryLabel[0]))

	return data, labels

def stdPersonXYZ(data, numberOfTests, timeScores):
	for i in range(0,int(numberOfTests)):
		print("test", i)
		offset = 0
		offset2 = 0
		for j in range (0, 25):
			dataPointsX = []
			dataPointsY = []
			dataPointsZ = []
			for k in range (offset, offset + 3*math.floor(maxEntries/2)):
				if (k < offset + (timeScores[i]//2)*3):
					if (k%3 == 0):
						dataPointsX.append(data[i][k])
					elif (k%3 == 1):
						dataPointsY.append(data[i][k])
					elif (k%3 == 2):
						dataPointsZ.append(data[i][k])
				else:
					k =  offset2 + (timeScores[i]//2)*3
			'''
			for k in range (offset2, offset2 + 3*math.floor(maxEntries/2)):
				if (k < offset2 + (timeScores[i]//2)*3):
					if (k%3 == 0):
						data[i][k] = (data[i][k]- stat.mean(dataPointsX))/stat.stdev(dataPointsX)
					elif (k%3 == 1):
						data[i][k] = (data[i][k]-stat.mean(dataPointsY))/stat.stdev(dataPointsY)
					elif (k%3 == 2):
						data[i][k] = (data[i][k]-stat.mean(dataPointsZ))/stat.stdev(dataPointsZ)
				else:
					k =  offset2 + (timeScores[i]//2)*3
			'''
			for k in range (offset2, offset2 + 3*math.floor(maxEntries/2)):
				if (k < offset2 + (timeScores[i]//2)*3):
					if (k%3 == 0):
						data[i][k] = (data[i][k]- min(dataPointsX))/(max(dataPointsX) - min(dataPointsX))
					elif (k%3 == 1):
						data[i][k] = (data[i][k]-min(dataPointsY))/(max(dataPointsY) - min(dataPointsY))
					elif (k%3 == 2):
						data[i][k] = (data[i][k]-min(dataPointsZ))/(max(dataPointsZ) - min(dataPointsZ))
				else:
					k =  offset2 + (timeScores[i]//2)*3

			offset = offset + 3*math.floor(maxEntries/2)
			offset2 = offset2 + 3*math.floor(maxEntries/2)

	newDir = "D:\\CMU\\CMU_HCII_REU\\ExerciseDetection\\stdData"
	if not (os.path.exists(newDir)):
		os.makedirs(newDir)

	for i in range( 0, int(numberTests)):
		newDir = "D:\\CMU\\CMU_HCII_REU\\ExerciseDetection\\stdData\\test" + str(i)
		if not (os.path.exists(newDir)):
			os.makedirs(newDir)
		offset2 = 0
		for j in range (0, 25):
			bp = math.floor(j/(3 * math.floor(maxEntries//2)))
			resultsFileL = open(newDir + "\\Position_" + file_names_super[int(j)] + ".csv", "a+")
			for k in range (offset2, offset2 + 3*math.floor(maxEntries/2)):
				if (k < offset2 + (timeScores[i]//2)*3 and k %3 == 0):
					resultsFileL.write(str(data[i][k]) + "," + str(data[i][k+1]) + "," + str(data[i][k+2]) + '\n')
				else:
					k =  offset2 + (timeScores[i]//2)*3
			offset2 = offset2 + 3*math.floor(maxEntries/2)

	return data

maxEntries, timeScores = calcMaxEntries()
velocityData, labels = extractData()
velocityData = stdPersonXYZ(velocityData,numberTests, timeScores)
