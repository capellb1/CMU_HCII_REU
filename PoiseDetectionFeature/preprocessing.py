import os
import random
import numpy as np
import math
import statistics as stat

DATA_FOLDER = 'DataWindow'

class preprocess:
	
	all_labels = []
	feature_vector = []
	std_feature_vector = []
	
	def __init__(self, data):
		self.data = data
		
		#Calculate Center of Mass
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

		#Calculate Distance from Wrist to Ankle
		handX = (self.data[30] + self.data[45])/2
		handY = (self.data[31] + self.data[46])/2
		handZ = (self.data[32] + self.data[47])/2
		footX = (self.data[60] + self.data[69])/2
		footY = (self.data[61] + self.data[70])/2
		footZ = (self.data[62] + self.data[71])/2
		self.wrist_to_ankle = math.sqrt((handX-footX)*(handX-footX) + (handY-footY)*(handY-footY) + (handZ-footZ)*(handZ-footZ)) 
		
		#Calculate Distance from Chest to Knee	
		chestX = self.data[9]
		chestY = self.data[10]
		chestZ = self.data[11]
		kneeX = (self.data[57] + self.data[66])/2
		kneeY = (self.data[58] + self.data[67])/2
		kneeZ = (self.data[59] + self.data[68])/2
		self.chest_to_knee = math.sqrt((chestX-kneeX)*(chestX-kneeX) + (chestY-kneeY)*(chestY-kneeY) + (chestZ-kneeZ)*(chestZ-kneeZ)) 

		#Calculate normal vector from chest
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


	def add_feat(self):
		#add a single feature vector frame to the exercise's list
		feat_vec = [self.comX, self.comY, self.comZ, self.wrist_to_ankle, self.chest_to_knee, self.chestX, self.chestY, self.chestZ]
		self.feature_vector.append(feat_vec)

	def add_std_ft(self, feature, label):
		#add entire exercise worth of standardized data to the overall list
		std_feat_vec = feature
		for i in range(0, len(std_feat_vec)):
			self.std_feature_vector.append(std_feat_vec[i])
			self.all_labels.append(label)
	
	def cl_feat(self):
		#used to reset feature vector at the end of each exercise
		self.feature_vector = []


def extract_data(i):
	
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

	line_ct = 0


	sample = open(dirname + "\\" + DATA_FOLDER + "\\test" + str(i)+ "\\Position_" + file_names[0])
	for line in sample:
		line_ct = line_ct + 1

	for line in open(dirname + "\\" + DATA_FOLDER + "\\test" + str(i)+ "\\label.csv"):
		temporaryLabel = line.split()
		labels.append(str(temporaryLabel[0]))


	for k in range(0, line_ct):
	
		line_data = []

		for j in range(0,25):
			fr = open(dirname + "\\" + DATA_FOLDER + "\\test" + str(i)+ "\\Position_" + file_names[j])
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

def store_std(data, labels):
	dirname = os.path.realpath('.')
	new_file_std = open(dirname + '\\std_data_oirignal_recalc.csv', 'w+') #create if doesnt already exist
	print(len(data) == len(labels))
	for i in range(0,len(data)):
		data_str = str(data[i]).replace('\'', '')
		data_str = data_str.replace('[', '')
		data_str = data_str.replace(']', '')
		new_file_std.write(labels[i] + ',' + data_str + '\n')

	print("STD Stored")

def std(features):
	
	feature_vector = []
	synthetic_feat = [[],[],[],[],[],[],[],[]]

	for i in range(0,8):
		for j in range(0,len(features)):
			if i == 0:
				synthetic_feat[i].append(features[j][i])
			if i == 1:
				synthetic_feat[i].append(features[j][i])
			if i == 2:
				synthetic_feat[i].append(features[j][i])
			if i == 3:
				synthetic_feat[i].append(features[j][i])
			if i == 4:
				synthetic_feat[i].append(features[j][i])
			if i == 5:
				synthetic_feat[i].append(features[j][i])
			if i == 6:
				synthetic_feat[i].append(features[j][i])
			if i == 7:
				synthetic_feat[i].append(features[j][i])
	
	mean  = []
	stdev = []
	std_feat = [[],[],[],[],[],[],[],[]]
	for i in range(0,8):
		mean.append(stat.mean(synthetic_feat[i]))
		stdev.append(stat.stdev(synthetic_feat[i]))
		for j in range(0,len(synthetic_feat[0])):			
			if i == 0:
				std_feat[i].append((synthetic_feat[i][j]-mean[i])/(stdev[i]))
			if i == 1:
				std_feat[i].append((synthetic_feat[i][j]-mean[i])/(stdev[i]))
			if i == 2:
				std_feat[i].append((synthetic_feat[i][j]-mean[i])/(stdev[i]))
			if i == 3:
				std_feat[i].append((synthetic_feat[i][j]-mean[i])/(stdev[i]))
			if i == 4:
				std_feat[i].append((synthetic_feat[i][j]-mean[i])/(stdev[i]))
			if i == 5:
				std_feat[i].append((synthetic_feat[i][j]-mean[i])/(stdev[i]))
			if i == 6:
				std_feat[i].append((synthetic_feat[i][j]-mean[i])/(stdev[i]))
			if i == 7:
				std_feat[i].append((synthetic_feat[i][j]-mean[i])/(stdev[i]))

	for j in range(0,len(std_feat[0])):
		feature_vector.append([std_feat[0][j],std_feat[1][j],std_feat[2][j],std_feat[3][j],std_feat[4][j],std_feat[5][j],std_feat[6][j],std_feat[7][j]])

	return feature_vector

def main(argv = None):

	#determine number of tests in dataset
	dirname = os.path.realpath('.')
	filename = dirname + '\\' + DATA_FOLDER + '\\TestNumber.txt'
	numberTestFiles = open(filename,"r")
	numberTests = numberTestFiles.read()	
	numTests = int(numberTests)

	for i in range(0, numTests):
		#extract all data for a single test
		data, labels = extract_data(i)
		for j in range(0, len(data)):
			#isolate a single frame
			for k in range(0,len(data[j])):
				#Float cast every value in the frame
				data[j][k] = float(data[j][k])

			feat = preprocess(data[j]) #preprocess frame
			feat.add_feat() #add to overall feature vector

		std_feature_vector = std(feat.feature_vector) #Standardize for entire exercise
		feat.add_std_ft(std_feature_vector, labels[j]) #Add to overall std feature vector
		feat.cl_feat()#reset feature vector at end of exercise

	#shuffle
	combined = list(zip(feat.std_feature_vector, feat.all_labels))
	random.shuffle(combined)
	std_feature_vector, all_labels = zip(*combined)
	store_std(std_feature_vector, all_labels)

	


#needed in order to call main
if __name__ == '__main__':
	main()





#data input into model is (feat.all_labels, feat.feature_vector)