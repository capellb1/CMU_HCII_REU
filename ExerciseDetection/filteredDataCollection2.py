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
numberTests = 1
bodySize = 25
numSection = 1

def calcMaxEntries():
	maxEntries = 0
	timeScores = []

	for i in range(0,int(numberTests)):			
		numEntries = 0
		for line in open(dirname + "\\selectedData\\test" + str(i) + "\\" + "Velocity_" + file_names_super[0]):
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
	print(len(data[0]))
	#enables downsampling by 50%
	labels = []

	for i in range(0, int(numberTests)):

		for j in range(0,bodySize):
			sample = True


			k = j*math.floor(maxEntries//2)*3
			
			for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\Velocity_" + file_names[j]):
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


	'''
	for i in range( 0, len(data)):
		resultsFileL = open("C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\Visualization2\\data" + str(i) + ".csv", "a+")
		resultsFileL.write("new test" + str(i) + '\n')
		for j in range (0, len(data[0])):
			if (j%(3 * math.floor(maxEntries//2)) == 0):
				resultsFileL.write("New Bodypart:" + '\n')
			resultsFileL.write(str(data[i][j]) + '\n')
	'''
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

			for p in range (offset2, offset2 + 3*math.floor(maxEntries/2)):

				if (p < offset2 + (timeScores[i]//2)*3):

					if (k%3 == 0):
						data[i][p] = (data[i][p]- stat.mean(dataPointsX))/stat.stdev(dataPointsX)
					elif (k%3 == 1):
						data[i][p] = (data[i][p]-stat.mean(dataPointsY))/stat.stdev(dataPointsY)
					elif (k%3 == 2):
						data[i][p] = (data[i][p]-stat.mean(dataPointsZ))/stat.stdev(dataPointsZ)

			offset = offset + 3*math.floor(maxEntries/2)
			offset2 = offset2 + 3*math.floor(maxEntries/2)
	return data

maxEntries, timeScores = calcMaxEntries()
velocityData, labels = extractData()
velocityData = stdPersonXYZ(velocityData,numberTests, timeScores)
print("velocity:", velocityData)


bodySize = 25

#for i in range(0, int(numberTests)):
for i in range (0, 1):
	offset = 0
	newDir2 = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow\\test" + str(i)
	if not (os.path.exists(newDir2)):
		os.makedirs(newDir2)

	for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\label.csv"):
		temporaryLabel = line.split()
		temporaryLabel = temporaryLabel[0]
	
	resultsFileL = open("C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow\\test" + str(i) +"\\label.csv", "a+")
	resultsFileL.write(temporaryLabel)

	velocityAboveThreshold = np.zeros((25, math.floor(timeScores[i]/2)))

	for j in range(0,25):
		print("new body")
		l = 0
		for k in range(offset, offset + math.floor(maxEntries/2)*3):
			if k%(math.floor(maxEntries//2)*3) < math.floor(timeScores[i]/2)*3:
				if (k%3 == 0):
					dataPointScore = math.fabs(velocityData[i][k]) + math.fabs(velocityData[i][k+1]) + math.fabs(velocityData[i][k+2])
					if (dataPointScore > 3):
						velocityAboveThreshold[j][l] = 1
					else:
						velocityAboveThreshold[j][l] = 0
					l = l + 1

		offset = offset + math.floor(maxEntries/2)*3
		print(velocityAboveThreshold[j].tolist())

	maxConsecutive = np.zeros((25))
	consecutiveOnes = np.zeros((25))

	for j in range(0,25):
		for k in range (0, math.floor(timeScores[i]//2)):
			if (velocityAboveThreshold[j][k] == 1):
				consecutiveOnes[j] = consecutiveOnes[j] + 1
			else:
				if (maxConsecutive[j] < consecutiveOnes[j]):
					maxConsecutive[j] = consecutiveOnes[j]
				consecutiveOnes[j] = 0
	
	timeStamps = np.zeros((50))
	firstLoopFound = 0
	secondLoopFound = 0
	currentConsecutive = 0
	print("FINDING RANGES PER BODYPART")
	for j in range (0, 25):
		print("NEW BODYPART")
		for k in range (0, math.floor(timeScores[i]//2)):
			if(firstLoopFound == 0 and velocityAboveThreshold[j][k] == 0 and currentConsecutive >= 5):
				print("first loop bodypart", j,"timestamp", k)
				timeStamps[2*j] = k 
				firstLoopFound = 1
				currentConsecutive = 0

			elif(firstLoopFound == 0 and velocityAboveThreshold[j][k] == 0 and currentConsecutive != 5):
				currentConsecutive = 0

			elif(firstLoopFound == 0 and velocityAboveThreshold[j][k] == 1 ):			
				currentConsecutive = currentConsecutive + 1

			elif (firstLoopFound == 1 and velocityAboveThreshold[j][k] == 1 and currentConsecutive != 3):
				currentConsecutive = currentConsecutive + 1

			elif(firstLoopFound == 1 and velocityAboveThreshold[j][k] == 1 and currentConsecutive == 3):
				print("second loop bodypart", j,"timestamp", k)
				timeStamps[2*j + 1] = k
				firstLoopFound = 0
				currentConsecutive = 0
		firstLoopFound = 0
		currentConsecutive = 0

	print("FINDING FINAL RANGE")
	range1 = 10000000000
	range2 = -10000000000
	for j in range (0, 25):
		print("first loop", timeStamps[2*j], "second loop", timeStamps[2*j + 1])
		if (timeStamps[2*j]!=0 and timeStamps[2*j+1]!=0 and timeStamps[2*j] < range1):
			range1 = timeStamps[2*j]
		if (timeStamps[2*j]!=0 and timeStamps[2*j+1]!=0 and timeStamps[2*j+1] > range2):
			range2 = timeStamps[2*j+1]

	print("ranges", range1, range2)

	k = 0
	for j in range(0,bodySize):
		resultsFileP = open("C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow\\test" + str(i) + "\\Position_" + file_names[j], "a+")
		m = 0
		sample = True
		for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\Position_" + file_names[j]):
			if m >= range1 and m <= range2:
				if sample:
					row = line.split(',')
					coords = []
					for l in range(0,3):
						coords.append(row[l])
						k = k +1
						sample = False
					coords = coords[0] + "," + coords[1] + "," + coords[2] + "\n"
					resultsFileP.write(coords) 
				else:
					sample = True
			m = m + 1
		resultsFileV = open("C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow\\test" + str(i) + "\\Velocity_" + file_names[j], "a+")
		m = 0
		sample = True
		for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\Velocity_" + file_names[j]):
			if m >= range1 and m <= range2:
				if sample:
					row = line.split(',')
					coords = []
					for l in range(0,3):
						coords.append(row[l])
						k = k +1
						sample = False
					coords = coords[0] + "," + coords[1] + "," + coords[2] + "\n"
					resultsFileV.write(coords) 
				else:
					sample = True
			m = m + 1

		resultsFileT = open("C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow\\test" + str(i) + "\\Task_" + file_names[j], "a+")
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
					coords = coords[0] + "," + coords[1] + "," + coords[2] + "\n"
					resultsFileT.write(coords) 
				else:
					sample = True
			m = m + 1
		
print ("done")

