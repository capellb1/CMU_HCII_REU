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

Used originally to calculate the z score for all of the data one time. Incorrectly calculates the information based off of all participants
instead of a person by person basis.
'''

#Import Libraries
import math
import io
import statistics as stat
from shutil import copyfile

#to get rid of warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_FOLDER = "DataCollectionSample"

#Predetermined selection of Bodyparts (CHANGE FOR REFINEMENT)
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

dirname = os.path.realpath('.')
filename = dirname + '\\' + DATA_FOLDER + '\\TestNumber.txt'

numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()

def stdPos(timeScores):
	features = []
	for j in range(0,75):
		features.append([])

	#extract data
	for i in range(0, int(numberTests)):
			for j in range(0,25):
				for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+"\\Position_" + file_names[j]):
					row = line.split(',')
					for k in range(0,3):
						features[3*j+k].append(float(row[k]))

	#calculate average/stdev/z scores for all features
	for i in range(0,75):
		meanTemp = stat.mean(features[i])
		stdevTemp = stat.stdev(features[i])

		for j in range(0, len(features[i])):
			features[i][j] = (features[i][j] - meanTemp)/(stdevTemp)


	for i in range(0,int(numberTests)):
		for j in range(0,25):
			posfile = open(dirname + "\\stdData\\test" + str(i)+ "\\Position_" + file_names[j], "w+")
			for l in range(0, timeScores[i]):
					posfile.write(str(features[3*j][l]) + "," + str(features[3*j+1][l]) + "," + str(features[(3*j)+2][l]) + '\n')
			posfile.close()
		copyfile(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+"\\label.csv", dirname + "\\stdData\\test" + str(i)+"\\label.csv")

def stdVel(timeScores):
	features = []
	for j in range(0,75):
		features.append([])

	#extract data
	for i in range(0, int(numberTests)):
			for j in range(0,25):
				for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+"\\Velocity_" + file_names[j]):
					row = line.split(',')
					for k in range(0,3):
						features[3*j+k].append(float(row[k]))

	#calculate average/stdev/z scores for all features
	for i in range(0,75):
		meanTemp = stat.mean(features[i])
		stdevTemp = stat.stdev(features[i])

		for j in range(0, len(features[i])):
			features[i][j] = (features[i][j] - meanTemp)/(stdevTemp)


	for i in range(0,int(numberTests)):
		for j in range(0,25):
			velfile = open(dirname + "\\stdData\\test" + str(i)+ "\\Velocity_" + file_names[j], "w+")
			for l in range(0, timeScores[i]):
					velfile.write(str(features[3*j][l]) + "," + str(features[(3*j)+1][l]) + "," + str(features[(3*j)+2][l]) + '\n')

			velfile.close()			
		copyfile(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+"\\label.csv", dirname + "\\stdData\\test" + str(i)+"\\label.csv")

def main(argv = None):
	
	if not (os.path.exists(dirname + "\\stdData")):
		os.makedirs(dirname + "\\stdData")

	maxEntries = 0
	timeScores = []
	for i in range(0,int(numberTests)):
		
		if not (os.path.exists(dirname + "\\stdData\\test" + str(i))):
			os.makedirs(dirname + "\\stdData\\test" + str(i))
		
		numEntries = 0
		for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i) + "\\Position_" + file_names[0]):
			numEntries = numEntries + 1
		if numEntries > maxEntries:
			maxEntries = numEntries	
		timeScores.append(numEntries)
		features = []

	#stdPos(timeScores)
	#stdVel(timeScores)
	copyfile(dirname + "\\"+ DATA_FOLDER +"\\TestNumber.txt", dirname + "\\stdData\\TestNumber.txt")

	labels = []
	for i in range(0,int(numberTests)):
		for line in open(dirname + "\\stdData\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			labels.append((str(temporaryLabel[0]),i))

	print(labels)
#needed in order to call main
if __name__ == '__main__':
	main()



			