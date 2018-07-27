import os
import shutil
import math
import io
import statistics as stat

DATA_FOLDER = "stdData"

#determine num of files in dataset
dirname = os.path.realpath('.')
filename = dirname + '\\stdData\\TestNumber.txt'
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()

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

def main(argv = None):
	if (os.path.exists(dirname + "\\stdData2")):
		shutil.rmtree(dirname + "\\stdData2")

	labels = []

	#Collect all the labels
	for i in range(0, int(numberTests)):
		for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			labels.append((str(temporaryLabel[0]),i))

	print("Labels:", labels)
	#parse list for desired examples
	removeIndex = [label[1] for label in labels if (label[0] == "Cat" or label[0] == "Supine" or label[0] == "Trunk" or  label[0] == "Pretzel" or label[0] == "OOV")]
	LabelsIndex = [label[1] for label in labels]
	print("Remove Index:", removeIndex)
	print("Labels Index:", LabelsIndex)

	#move desired exercises out of the folder
	shutil.copytree(dirname + "\\" + DATA_FOLDER, dirname + "\\stdData2")
	for i in range(0, len(removeIndex)):
		shutil.rmtree(dirname + "\\stdData2\\test" + str(removeIndex[i]))
		LabelsIndex.remove(removeIndex[i])

	print("Updated Labels Index:", LabelsIndex)

	#update the numTests file
	numberTestFiles = open(dirname + "\\stdData2\\TestNumber.txt","w+")
	print("New File Size: ", len(LabelsIndex))
	numberTestFiles.write(str(len(LabelsIndex)))
	numberTestsSel = len(LabelsIndex)

	#update the numbering on remaining files
	for i in range(0, len(LabelsIndex)):
		os.rename((dirname + "\\stdData2\\test" + str(LabelsIndex[i])), (dirname + "\\stdData2\\test"+ str(i)))

	

	maxEntries = 0
	timeScores = []
	for i in range(0,int(numberTests)):
		numEntries = 0
		for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i) + "\\Position_" + file_names[0]):
			numEntries = numEntries + 1
		if numEntries > maxEntries:
			maxEntries = numEntries	
		timeScores.append(numEntries)
		features = []

	print("Data Stored in: ", dirname, "\\stdData2")

if __name__ == '__main__':
	main()