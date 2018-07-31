#Import Libraries
import math
import io

#to get rid of warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Tensorflow and Data Processing Library
import tensorflow as tf
from tensorflow.python.data import Dataset
import numpy as np
import pandas as pd
import math 
import statistics as stat

class PreProcessing:
	def __init__(self, folderName):
		self.DATA_FOLDER = folderName
		self.dirname = os.path.realpath('.')

	numberTests = 0 
	maxEntries = 0
	timeScores = []

	file_names =[
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
		'FootLeft.csv'
		]
	def calcNumTests(self):
		filename = self.dirname + '\\' + self.DATA_FOLDER + '\\TestNumber.txt'

		numberTestFiles = open(filename, "r")
		self.numberTests = numberTestFiles.read()

	def calcMaxEntries(self):
		for i in range (0, int(self.numberTests)):
			numEntries = 0

			for line in open(self.dirname + "\\" + self.DATA_FOLDER + "\\test" + str(i) + "\\Position_" + self.file_names[0]):
				numEntries = numEntries + 1
			if numEntries > self.maxEntries:
				self.maxEntries = numEntries
			self.timeScores.append(numEntries)

	def extractData(self):
		'''
			Moves data from the text files into flattened arrays.
			Each time stamp is a single row and has a corresponding event label
				[Arm1xyz, Head1xyz, Foot1xyz, ...] EVENT 1
				[Arm2xyz, Head2xyz, Foot2xyz, ...] EVENT 2
			
			Parameters: None
			Returns:
				nparray shuffledlabels
				nparray shuffledData
		'''
		#average
		data =  np.empty((sum(self.timeScores), int(25*3)))
		sample = True

		numTimeStamps = 0
		labels = []
		edges = []
		c=0
		for i in range(0, int(self.numberTests)):
			#Determine the number of time stamps in this event
			w=0

			for l in range(numTimeStamps,numTimeStamps+(self.timeScores[i])):
				k=0
				for j in range(0, 25):
					fp = open(self.dirname + "\\"+ self.DATA_FOLDER +"\\test" + str(i)+ "\\Position_" + self.file_names[j] )
					for n, line in enumerate(fp):
						if n == w:
							row = line.split(',')
							for m in range(0,3):
								data[l][k]= row[m]
								k = k + 1
				
					
				for line in open(self.dirname + "\\"+ self.DATA_FOLDER +"\\test" + str(i)+ "\\label.csv"):
					temporaryLabel = line.split()
					labels.append(str(temporaryLabel[0]))

				w=w+1	
			
			edges.append(numTimeStamps)
			
			numTimeStamps = (self.timeScores[i]) + numTimeStamps

		fp.close()

		#shuffle the data
		self.shuffledData = np.empty(data.shape, dtype=data.dtype)
		self.shuffledLabels = labels
		permutation = np.random.RandomState(seed=42).permutation(len(labels))
		for old_index, new_index in enumerate(permutation):
			self.shuffledData[new_index] = data[old_index]
			self.shuffledLabels[new_index] = labels[old_index]

		self.shuffledLabels = np.asarray(self.shuffledLabels)
		self.featuredData = np.empty((sum(self.timeScores), 8))

	def CenterOfMass(self):
		for i in range (0, len(self.featuredData)):
			sumX = 0
			sumY = 0
			sumZ = 0
			for j in range (0, len(self.featuredData[i])):
				if (j%3 == 0):
					sumX = sumX + self.shuffledData[i][j]
				if (j%3 == 1):
					sumY = sumY + self.shuffledData[i][j]
				if (j%3 == 2):
					sumZ = sumZ + self.shuffledData[i][j]		
			self.featuredData[i][0] = float((sumX)/25)
			self.featuredData[i][1] = float((sumY)/25)
			self.featuredData[i][2] = float((sumZ)/25)


	def DistanceHandFoot(self):
		for i in range (0, len(self.featuredData)):
			handX = self.shuffledData[i][30] + self.shuffledData[i][45]
			handY = self.shuffledData[i][31] + self.shuffledData[i][46]
			handZ = self.shuffledData[i][32] + self.shuffledData[i][47]
			footX = self.shuffledData[i][60] + self.shuffledData[i][69]
			footY = self.shuffledData[i][61] + self.shuffledData[i][70]
			footZ = self.shuffledData[i][62] + self.shuffledData[i][71]

			self.featuredData[i][3] = math.sqrt((handX-footX)*(handX-footX) + (handY-footY)*(handY-footY) + (handZ-footZ)*(handZ-footZ)) 

	def DistanceChestKnee(self):
		for i in range (0, len(self.featuredData)):
			chestX = self.shuffledData[i][9]
			chestY = self.shuffledData[i][10]
			chestZ = self.shuffledData[i][11]
			kneeX = self.shuffledData[i][57] + self.shuffledData[i][66]
			kneeY = self.shuffledData[i][58] + self.shuffledData[i][67]
			kneeZ = self.shuffledData[i][59] + self.shuffledData[i][68]
			self.featuredData[i][4] = math.sqrt((chestX-kneeX)*(chestX-kneeX) + (chestY-kneeY)*(chestY-kneeY) + (chestZ-kneeZ)*(chestZ-kneeZ)) 

	def NormalizingChest(self):
		for i in range (0, len(self.featuredData)):
			ShoulderLeft = np.zeros((3))
			ShoulderRight = np.zeros((3)) 
			Chest = np.zeros((3))

			ShoulderLeft[0] = self.shuffledData[i][18]
			ShoulderLeft[1] = self.shuffledData[i][19]
			ShoulderLeft[2] = self.shuffledData[i][20]

			ShoulderRight[0] = self.shuffledData[i][15]
			ShoulderRight[1] = self.shuffledData[i][16]
			ShoulderRight[2] = self.shuffledData[i][17]

			Chest[0] = self.shuffledData[i][9]
			Chest[1] = self.shuffledData[i][10]
			Chest[2] = self.shuffledData[i][11]

			Vector1 = Chest - ShoulderLeft
			Vector2 = Chest - ShoulderRight

			Normal = np.cross(Vector1, Vector2)

			self.featuredData[i][5] = Normal[0]
			self.featuredData[i][6] = Normal[1]
			self.featuredData[i][7] = Normal[2]

Data = PreProcessing('TwoDataCollectionSample')
Data.calcNumTests()
Data.calcMaxEntries()
Data.extractData()
Data.CenterOfMass()
Data.DistanceHandFoot()
Data.DistanceChestKnee()
Data.NormalizingChest()