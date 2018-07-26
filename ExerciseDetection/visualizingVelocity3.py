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
		newDir = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\stdData\\test" + str(i)
		numEntries = 0
		for line in open(newDir + "\\Velocity_" + file_names_super[int(0)] + ".csv"):
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
	#enables downsampling by 50%
	labels = []

	for i in range(0, int(numberTests)):
		newDir = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\stdData\\test" + str(i)
		for j in range(0,bodySize):

			k = j*maxEntries*3
			
			for line in open(newDir + "\\Velocity_" + file_names_super[int(j)] + ".csv"):
				if k < j*maxEntries*3 + maxEntries*3:
					row = line.split(',')
					for l in range(0,3):
						data[i][k] = row[l]
						k = k +1
			
		#seperate the label from the name and event number stored within the label.csv file(s)
		for line in open(newDir + "\\label.csv"):
			temporaryLabel = line.split()
			labels.append(str(temporaryLabel[0]))

	return data, labels


def draw(velocity):	
	'''
		Creates graphs that plot the accuracy of predictions for every frame in an action (Blue). Below, on 
		the same image, a graph representing the number of tasks detected across each body part for 
		that each frame (Red). This occurs for each example in the data.
		
		The function will also output a cumulative histogram for each action with more
		than one example (Green). This cumulative histogram overlays the accuracy of every example of a given
		exercise.
	'''
	start = 0	
	offset = 0

	newDir = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\Visualization3"
	if not (os.path.exists(newDir)):
		os.makedirs(newDir)
	
	totalVelocityData = []
	for i in range (0, int(numberTests)):
		offset = 0
		velocityData = np.zeros((25, timeScores[i]))
		maxZ = 0
		for j in range(0,25):
			print("finding bodypart data")
			l = 0
			for k in range(offset, offset + maxEntries*3):
				if k%(maxEntries*3) < timeScores[i]*3:
					if (k%3 == 0):
						velocityData[j][l] = math.fabs(velocity[i][k]) + math.fabs(velocity[i][k+1]) + math.fabs(velocity[i][k+2])
						if (velocityData[j][l] > maxZ):
							maxZ = velocityData[j][l]
						l = l + 1


			offset = offset + maxEntries*3

		newDir2 = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\Visualization3\\test" + str(i)
		if not (os.path.exists(newDir2)):
			os.makedirs(newDir2)

		graphDataX = []
		for j in range(0, timeScores[i]):
			graphDataX.append(j)
		
		print("about to print")
		for g in range (0, 25):
			n, bins, patches = plt.hist(velocityData[g], 'auto', normed=1, facecolor='green', alpha=0.75)

			plt.xlabel('Velocity Z Scores')
			plt.ylabel('Number of Frames')
			plt.title('Velocity Z Score vs Frames')
			plt.axis('auto')
			plt.grid(True)
			plt.savefig(newDir2 +"\\bodypartHisto" + str(g) + ".png")
			plt.close()

			width = .99
			plt.bar(graphDataX, velocityData[g], width, facecolor='blue')
			plt.savefig(newDir2 +"\\bodypartBar" + str(g) + ".png")
			plt.close()
		
		totalVelocityData.append(velocityData)

maxEntries, timeScores = calcMaxEntries()
velocityData, labels = extractData()
draw(velocityData)
