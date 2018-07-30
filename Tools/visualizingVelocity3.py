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

File that plots the velocity functions along with creating a histogram. Information used to verify window selection and illustrate
the information contained within the data.
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

print(os.getcwd())

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
		newDir = dirname + "\\stdData\\test" + str(i)
		numEntries = 0
		for line in open("D:\\CMU\\CMU_HCII_REU\\ExerciseDetection\\stdData\\test" + str(i) +"\\Velocity_Head.csv.csv"):
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
		newDir = dirname + "\\stdData\\test" + str(i)
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


def draw(velocity , thresh):	
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

	newDir = dirname + "\\Visualization3"
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

		newDir2 = dirname + "\\Visualization3\\test" + str(i)
		if not (os.path.exists(newDir2)):
			os.makedirs(newDir2)

		graphDataX = []
		for j in range(0, timeScores[i]):
			graphDataX.append(j)
		'''
		print("about to print")
		for g in range (0, 25):
			n, bins, patches = plt.hist(velocityData[g], 'auto', normed=1, facecolor='#800000', alpha=0.75)
			bodyName = file_names_super[g]
			bodyName = bodyName[:-4]
			plt.xlabel('Velocity Z Scores')
			plt.ylabel('Proportion of Frames')
			plt.title('Trial' + str(i) + ': ' + str(bodyName) + ' Velocity Histogram')
			plt.axis('auto')
			plt.grid(True)
			plt.savefig(newDir2 +"\\bodypartHisto" + str(g) + ".png")
			plt.close()

			width = .99
			plt.bar(graphDataX, velocityData[g], width, facecolor='#800000')
			plt.xlabel('Frame')
			plt.ylabel('Cumulative Velocity Across All Axes')
			plt.title('Trial' + str(i) + ': ' + str(bodyName) + ' Velocity Plot')
			plt.grid(True)
			plt.savefig(newDir2 +"\\bodypartBar" + str(g) + ".png")
			plt.close()
		'''
		totalVelocityData.append(velocityData)

	width = .99
	fig, ax = plt.subplots()
	print(len(totalVelocityData[44][0]), len(totalVelocityData[46][0]), len(totalVelocityData[48][0]))
	ax.bar(range(0,len(totalVelocityData[48][0])), totalVelocityData[48][0], width, facecolor='#F2B8C6', label = "Trial 48: Mermaid") #Crepe
	ax.bar(range(0,len(totalVelocityData[44][0])), totalVelocityData[44][0], width, facecolor='#800000', label = "Trial 44: Sumo") #Maroon
	ax.plot(range(0,len(totalVelocityData[44][0])), ([3.6]*len(totalVelocityData[44][0])), '--', label = "Threshold", color = '#FDA50F') #Orange
	plt.legend()
	plt.xlabel('Frame')
	plt.ylabel('Cumulative Velocity Across All Axes')
	plt.title('Velocity of Head Across Multiple Exercises')
	plt.grid(True)
	plt.savefig(newDir +"\\overlapBar.png")
	plt.close()

	width = .99
	fig, ax = plt.subplots()
	
	ax.bar(range(0,len(totalVelocityData[0][0])), totalVelocityData[0][0], width, facecolor='#800000', label = "Trial 0: Y") #Crepe
	ax.plot(range(0,len(thresh[0])), thresh[0], '--', label = "Transition", color = '#FDA50F')
	
	plt.ylabel('Cumulative Velocity Across All Axes')
	plt.legend(loc = 2)
	plt.xlabel('Frame')
	plt.title('Velocity of Head')
	plt.grid(True)
	plt.savefig(newDir +"\\section.png")
	plt.close()

maxEntries, timeScores = calcMaxEntries()
velocityData, labels = extractData()

offset = 0
velocityAboveThreshold = np.zeros((25, timeScores[0]))

for j in range(0,25):
	l = 0
	for k in range(offset, offset + maxEntries*3):
		if k%(maxEntries*3) < timeScores[0]*3:
			if (k%3 == 0):
				dataPointScore = math.fabs(velocityData[0][k]) + math.fabs(velocityData[0][k+1]) + math.fabs(velocityData[0][k+2])
				if (dataPointScore > 3.6):
					velocityAboveThreshold[j][l] = 8
				else:
					velocityAboveThreshold[j][l] = 0
				l = l + 1

	offset = offset + maxEntries*3

draw(velocityData, velocityAboveThreshold)
