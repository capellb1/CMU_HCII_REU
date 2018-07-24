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

def calcMaxEntries():
	maxEntries = 0
	timeScores = []
	for i in range(0,int(numberTests)):
		numEntries = 0
		for line in open(dirname + "\\selectedData\\test" + str(i) + "\\" + "velocity" + "_" + file_names_super[0]):
			numEntries = numEntries + 1
		if numEntries > maxEntries:
			maxEntries = numEntries	
		timeScores.append(numEntries)
	
	return maxEntries, timeScores

def extractData():
	'''
		Moves data from the text files into flattened arrays.
		Each time stamp is a single row and has a corresponding event label
			[Arm1xyz, Head1xyz, Foot1xyz, ...] EVENT 10
			[Arm2xyz, Head2xyz, Foot2xyz, ...] EVENT 2
		
		Parameters: None
		Returns:
			nparray labels
			nparray Data
	'''
	maxTime = 0
	for i in range (0 ,int(numberTests)):
		maxTime = maxTime + timeScores[i]//2

	data =  np.empty((maxTime, int(25*3)))
	sample = True
	numTimeStamps = 0
	labels = []
	c=0

	for i in range(0, int(numberTests)):
		#Determine the number of time stamps in this event
		w=0
		for l in range(numTimeStamps,numTimeStamps+timeScores[i]//2):
			k=0
			h=0
			for j in range(0, 25):
				fp = open(dirname + "\\selectedData\\test" + str(i)+ "\\Velocity_" + file_names[j])
				for n, line in enumerate(fp):
					if n == w:
						row = line.split(',')
						for m in range(0,3):
							data[l][k]= row[m]
							k = k + 1
				fp.close()
			for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\label.csv"):
				temporaryLabel = line.split()
				labels.append(str(temporaryLabel[0]))
				
			w=w+2	
					
		numTimeStamps = timeScores[i]//2 + numTimeStamps

	print(timeScores[0] + timeScores[1])
	print(data[0])
	print (data[1])
	print("date", len(data))
	print("lables", len(labels))
	print(labels[0], labels[0 + timeScores[0]//2])

	return data, labels

def stdPersonXYZ(data, numberOfTests, timeScores):
	print("original", data)
	for i in range(0,int(numberOfTests)):
		offset = 0
		for j in range (1, 1):
			dataPointsX = []
			dataPointsY = []
			dataPointsZ = []
			for k in range (offset, 3*j*timeScores[i]):
				if (k%3 == 0):
					dataPointsX.append(data[i][k])
				elif (k%3 == 1):
					dataPointsY.append(data[i][k])
				elif (k%3 == 2):
					dataPointsZ.append(data[i][k])

			for k in range (offset, 3*j*timeScores[i]):
				if (k%3 == 0):
					data[i][k] = (data[i][k]- stat.mean(dataPointsX))/stat.stdev(dataPointsX)
				elif (k%3 == 1):
					data[i][k] = (data[i][k]-stat.mean(dataPointsY))/stat.stdev(dataPointsY)
				elif (k%3 == 2):
					data[i][k] = (data[i][k]-stat.mean(dataPointsZ))/stat.stdev(dataPointsZ)
			offset = offset + 3*j*timeScores[i]
			print("meanX", stat.mean(dataPointsX))
			print("stdX", stat.stdev(dataPointsX))

	print("z-scores:", data)
	return data

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

	newDir = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\Visualization2"
	if not (os.path.exists(newDir)):
		os.makedirs(newDir)
	
	totalVelocityData = []
	for i in range (0, int(numberTests)):
		velocityData = np.zeros((25, timeScores[i]//2))
		graphDataX = []

		for j in range (0, timeScores[i]//2):

			for k in range ( 0, 25):

				for l in range (0, 3):

					velocityData[k][j] = velocityData[k][j] + math.fabs(velocity[offset+j][3*k + l])
			graphDataX.append(j)
		offset = timeScores[i]//2

		newDir2 = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\Visualization2\\test" + str(i)
		if not (os.path.exists(newDir2)):
			os.makedirs(newDir2)
		
		print("about to print")
		for g in range (0, 25):
			n, bins, patches = plt.hist(velocityData[g], 'auto', normed=1, facecolor='green', alpha=0.75)

			plt.xlabel('Velocity Z Scores')
			plt.ylabel('Number of Frames')
			plt.title('Velocity Z Score vs Frames')
			plt.axis('auto')
			plt.grid(True)
			plt.savefig(newDir2 +"\\bodypart" + str(g) + ".png")
			plt.close()
		
		totalVelocityData.append(velocityData)

	maxY = 0
	maxSeated = 0
	maxSumo = 0
	maxMermaid = 0
	maxTowel = 0
	maxWall = 0
	maxOOV = 0

	offset = -1
	for i in range (0, int(numberTests)):
		if (labels[offset + timeScores[i]//2].lower() == 'y'):
			if timeScores[i]//2 > maxY:
				maxY = timeScores[i]//2
			print("y")
			
		elif (labels[offset + timeScores[i]//2].lower() == 'seated'):
			if timeScores[i]//2 > maxSeated:
				maxSeated = timeScores[i]//2
			print("Seated")
			
		elif (labels[offset + timeScores[i]//2].lower() == 'sumo'):
			if timeScores[i]//2 > maxSumo:
				maxSumo = timeScores[i]//2
			print("sumo")
			
		elif (labels[offset + timeScores[i]//2].lower() == 'mermaid'):
			if timeScores[i]//2 > maxMermaid:
				maxMermaid = timeScores[i]//2
			print("mermaid")
			
		elif (labels[offset + timeScores[i]//2].lower() == 'towel'):
			if timeScores[i]//2 > maxTowel:
				maxTowel = timeScores[i]//2
			print("towel")
			
		elif (labels[offset + timeScores[i]//2].lower() == 'wall'):
			if timeScores[i]//2 > maxWall:
				maxWall = timeScores[i]//2
			print("wall")
			
		elif (labels[offset + timeScores[i]//2].lower() == 'oov'):
			if timeScores[i]//2 > maxOOV:
				maxOOV = timeScores[i]//2
			print("oov")
			
		else:
			print("random", labels[i])
		offset = offset + timeScores[i]//2

	YVelocityGraph = np.zeros((25, maxY))
	SeatedVelocityGraph = np.zeros((25, maxSeated))
	SumoVelocityGraph = np.zeros((25, maxSumo))
	MermaidVelocityGraph = np.zeros((25, maxMermaid))
	TowelVelocityGraph = np.zeros((25, maxTowel))
	WallVelocityGraph = np.zeros((25, maxWall))
	OOVVelocityGraph = np.zeros((25, maxOOV))

	offset = -1 
	for i in range (0, int(numberTests)):
		if (labels[offset + timeScores[i]//2].lower() == 'y'):
			if timeScores[i]//2 > maxY:
				maxY = timeScores[i]//2
			print("y")
			for j in range (0, timeScores[i]//2):
				for k in range (0, 25):
					YVelocityGraph[k][j] = YVelocityGraph[k][j] + totalVelocityData[i][k][j]
		
		elif (labels[offset + timeScores[i]//2].lower() == 'seated'):
			if timeScores[i]//2 > maxSeated:
				maxSeated = timeScores[i]//2
			print("Seated")
			for j in range (0, timeScores[i]//2):
				for k in range (0, 25):
					SeatedVelocityGraph[k][j] = SeatedVelocityGraph[k][j] + totalVelocityData[i][k][j]
		
		elif (labels[offset + timeScores[i]//2].lower() == 'sumo'):
			if timeScores[i]//2 > maxSumo:
				maxSumo = timeScores[i]//2
			print("sumo")
			for j in range (0, timeScores[i]//2):
				for k in range (0, 25):
					SumoVelocityGraph[k][j] = SumoVelocityGraph[k][j] + totalVelocityData[i][k][j]
		
		elif (labels[offset + timeScores[i]//2].lower() == 'mermaid'):
			if timeScores[i]//2 > maxMermaid:
				maxMermaid = timeScores[i]//2
			print("mermaid")
			for j in range (0, timeScores[i]//2):
				for k in range (0, 25):
					MermaidVelocityGraph[k][j] = MermaidVelocityGraph[k][j] + totalVelocityData[i][k][j]
		
		elif (labels[offset + timeScores[i]//2].lower() == 'towel'):
			if timeScores[i]//2 > maxTowel:
				maxTowel = timeScores[i]//2
			print("towel")
			for j in range (0, timeScores[i]//2):
				for k in range (0, 25):
					TowelVelocityGraph[k][j] = TowelVelocityGraph[k][j] + totalVelocityData[i][k][j]
		
		elif (labels[offset + timeScores[i]//2].lower() == 'wall'):
			if timeScores[i]//2 > maxWall:
				maxWall = timeScores[i]//2
			print("wall")
			for j in range (0, timeScores[i]//2):
				for k in range (0, 25):
					WallVelocityGraph[k][j] = WallVelocityGraph[k][j] + totalVelocityData[i][k][j]
		
		elif (labels[offset + timeScores[i]//2].lower() == 'oov'):
			if timeScores[i]//2 > maxOOV:
				maxOOV = timeScores[i]//2
			print("oov")
			for j in range (0, timeScores[i]//2):
				for k in range (0, 25):
					OOVVelocityGraph[k][j] = OOVVelocityGraph[k][j] + totalVelocityData[i][k][j]
		
		else:
			print("random", labels[i])
		offset = offset + timeScores[i]//2

	YDataX = []
	SeatedDataX = []
	SumoDataX = []
	MermaidDataX = []
	TowelDataX = []
	WallDataX = []
	OOVDataX = []

	for t in range (0, maxY):
		YDataX.append(t)

	for t in range (0, maxSeated):
		SeatedDataX.append(t)

	for t in range (0, maxSumo):
		SumoDataX.append(t)

	for t in range (0, maxMermaid):
		MermaidDataX.append(t)

	for t in range (0, maxTowel):
		TowelDataX.append(t)

	for t in range (0, maxWall):
		WallDataX.append(t)

	for t in range (0, maxOOV):
		OOVDataX.append(t)

	print (len(YDataX))
	print(len(YVelocityGraph[0]))
	if not (os.path.exists(dirname +"\\Visualization2\\Y\\")):
		os.makedirs(dirname +"\\Visualization2\\Y\\")
	for b in range (0, 25):
		width = .99
		plt.bar(YDataX, YVelocityGraph[b], width, facecolor='blue')
		plt.savefig(dirname +"\\Visualization2\\Y\\Bodypart" + str(b) + ".png")
		plt.close()
	
	if not (os.path.exists(dirname +"\\Visualization2\\Seated\\")):
		os.makedirs(dirname +"\\Visualization2\\Seated\\")
	for b in range (0, 25):
		width = .99
		plt.bar(SeatedDataX, SeatedVelocityGraph[b], width, facecolor='blue')
		plt.savefig(dirname +"\\Visualization2\\Seated\\Bodypart" + str(b) + ".png")
		plt.close()

	if not (os.path.exists(dirname +"\\Visualization2\\Sumo\\")):
		os.makedirs(dirname +"\\Visualization2\\Sumo\\")
	for b in range (0, 25):
		width = .99
		plt.bar(SumoDataX, SumoVelocityGraph[b], width, facecolor='blue')
		plt.savefig(dirname +"\\Visualization2\\Sumo\\Bodypart" + str(b) + ".png")
		plt.close()

	if not (os.path.exists(dirname +"\\Visualization2\\Mermaid\\")):
		os.makedirs(dirname +"\\Visualization2\\Mermaid\\")
	for b in range (0, 25):
		width = .99
		plt.bar(MermaidDataX, MermaidVelocityGraph[b], width, facecolor='blue')
		plt.savefig(dirname +"\\Visualization2\\Mermaid\\Bodypart" + str(b) + ".png")
		plt.close()
	
	if not (os.path.exists(dirname +"\\Visualization2\\Towel\\")):
		os.makedirs(dirname +"\\Visualization2\\Towel\\")
	for b in range (0, 25):
		width = .99
		plt.bar(TowelDataX, TowelVelocityGraph[b], width, facecolor='blue')
		plt.savefig(dirname +"\\Visualization2\\Towel\\Bodypart" + str(b) + ".png")
		plt.close()

	if not (os.path.exists(dirname +"\\Visualization2\\Wall\\")):
		os.makedirs(dirname +"\\Visualization2\\Wall\\")
	for b in range (0, 25):
		width = .99
		plt.bar(WallDataX, WallVelocityGraph[b], width, facecolor='blue')
		plt.savefig(dirname +"\\Visualization2\\Wall\\Bodypart" + str(b) + ".png")
		plt.close()

	if not (os.path.exists(dirname +"\\Visualization2\\OOV\\")):
		os.makedirs(dirname +"\\Visualization2\\OOV\\")
	for b in range (0, 25):
		width = .99
		plt.bar(OOVDataX, OOVVelocityGraph[b], width, facecolor='blue')
		plt.savefig(dirname +"\\Visualization2\\OOV\\Bodypart" + str(b) + ".png")
		plt.close()
	


maxEntries, timeScores = calcMaxEntries()
velocityData, labels = extractData()
velocityData = stdPersonXYZ(velocityData,numberTests, timeScores)
draw(velocityData)
