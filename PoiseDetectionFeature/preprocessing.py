import os
import random
import numpy as np
import math

class preprocess:
	
	all_labels = []
	feature_vector = []
	
	def __init__(self, data):
		self.data = data
		
		sumX = 0
		sumY = 0
		sumZ = 0
		for j in range (0, len(self.data)):
			#loops through all bodyparts in the frame
			if (j%3 == 0):
				sumX = sumX + self.data[j]
			if (j%3 == 1):
				sumY = sumY + self.data[j]
			if (j%3 == 2):
				sumZ = sumZ + self.data[j]		

		self.comX = float((sumX)/25)
		self.comY = float((sumY)/25)
		self.comZ = float((sumZ)/25)

		handX = (self.data[30] + self.data[45])/2
		handY = (self.data[31] + self.data[46])/2
		handZ = (self.data[32] + self.data[47])/2
		footX = (self.data[60] + self.data[69])/2
		footY = (self.data[61] + self.data[70])/2
		footZ = (self.data[62] + self.data[71])/2
		self.wrist_to_ankle = math.sqrt((handX-footX)*(handX-footX) + (handY-footY)*(handY-footY) + (handZ-footZ)*(handZ-footZ)) 
			
		chestX = self.data[9]
		chestY = self.data[10]
		chestZ = self.data[11]
		kneeX = (self.data[57] + self.data[66])/2
		kneeY = (self.data[58] + self.data[67])/2
		kneeZ = (self.data[59] + self.data[68])/2
		self.chest_to_knee = math.sqrt((chestX-kneeX)*(chestX-kneeX) + (chestY-kneeY)*(chestY-kneeY) + (chestZ-kneeZ)*(chestZ-kneeZ)) 

		ShoulderLeft = np.zeros((3))
		ShoulderRight = np.zeros((3)) 
		Chest = np.zeros((3))
		ShoulderLeft[0] = self.data[18]
		ShoulderLeft[1] = self.data[19]
		ShoulderLeft[2] = self.data[20]
		ShoulderRight[0] = self.data[15]
		ShoulderRight[1] = self.data[16]
		ShoulderRight[2] = self.data[17]
		Chest[0] = self.data[9]
		Chest[1] = self.data[10]
		Chest[2] = self.data[11]
		Vector1 = Chest - ShoulderLeft
		Vector2 = Chest - ShoulderRight
		Normal = np.cross(Vector1, Vector2)

		self.chestX = Normal[0]
		self.chestY = Normal[1]
		self.chestZ = Normal[2]

	def add_label(self, label):
		self.all_labels.append(label)
		self.label = label

	def add_feat(self):
		feat_vec = [self.comX, self.comY, self.comZ, self.wrist_to_ankle, self.chest_to_knee, self.chestX, self.chestY, self.chestZ]
		self.feature_vector.append(feat_vec)


def extract_data():
	
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
	'FootLeft.csv']

	data = []
	labels = []
	dirname = os.path.realpath('.')

	filename = dirname + '\\TwoDataCollectionSample\\TestNumber.txt'

	numberTestFiles = open(filename,"r")
	numberTests = numberTestFiles.read()	
	numTests = int(numberTests)

	for i in range(0,numTests):
		line_ct = 0

		sample = open(dirname + "\\TwoDataCollectionSample\\test" + str(i)+ "\\Position_" + file_names[0])
		for line in sample:
			line_ct = line_ct + 1

		for line in open(dirname + "\\TwoDataCollectionSample\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			labels.append(str(temporaryLabel[0]))


		for k in range(0, line_ct):
	
			line_data = []

			for j in range(0,25):
				fr = open(dirname + "\\TwoDataCollectionSample\\test" + str(i)+ "\\Position_" + file_names[j])
				for l, line in enumerate(fr):
					if l == k:
						#reads the (k-1)th line
						row = line.split(',')
						for m in range(0,3):
							line_data.append(row[m])

			data.append(line_data)
			labels.append(str(temporaryLabel[0]))

	#shuffle
	combined = list(zip(data, labels))
	random.shuffle(combined)
	data, labels = zip(*combined)

	return data, labels

def store_raw(data, labels):
	dirname = os.path.realpath('.')
	new_file = open(dirname + '\\raw_data.csv', 'w+') #create if doesnt already exist

	for i in range(0,len(data)):
		data_str = str(data[i]).replace('\'', '')
		data_str = data_str.replace('[', '')
		data_str = data_str.replace(']', '')
		new_file.write(labels[i] + ',' + data_str + '\n')

def store_calc(data, labels):
	dirname = os.path.realpath('.')
	new_file = open(dirname + '\\calc_data.csv', 'w+') #create if doesnt already exist

	for i in range(0,len(data)):
		data_str = str(data[i]).replace('\'', '')
		data_str = data_str.replace('[', '')
		data_str = data_str.replace(']', '')
		new_file.write(labels[i] + ',' + data_str + '\n')

def main(argv = None):

	data, labels = extract_data()
	store_raw(data, labels)
	for i in range(0, len(data)):
		
		for j in range(0,len(data[i])):
			data[i][j] = float(data[i][j])

		feat = preprocess(data[i])
		feat.add_feat()
		feat.add_label(labels[i])

	store_calc(feat.feature_vector, feat.all_labels)


#needed in order to call main
if __name__ == '__main__':
	main()





#data input into model is (feat.all_labels, feat.feature_vector)