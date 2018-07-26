import os
import shutil
import math
import io
import statistics as stat

DATA_FOLDER = "stdData"

#determine num of files in dataset
dirname = os.path.realpath('.')
filename = dirname + '\\' + DATA_FOLDER +'\\TestNumber.txt'
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
	if (os.path.exists(dirname + "\\selectedData")):
		shutil.rmtree(dirname + "\\selectedData")

	labels = []

	#Collect all the labels
	for i in range(0, int(numberTests)):
		for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			labels.append((str(temporaryLabel[0]),i))

	print("Labels:", labels)
	#parse list for desired examples
	removeIndex = [label[1] for label in labels if (label[0].lower() == "cat" or label[0].lower() == "supine" or label[0].lower() == "trunk" or  label[0].lower() == "pretzel" or label[0].lower() == "oov" )]
	LabelsIndex = [label[1] for label in labels]
	print("Remove Index:", removeIndex)
	print("Labels Index:", LabelsIndex)

	#move desired exercises out of the folder
	shutil.copytree(dirname + "\\" + DATA_FOLDER, dirname + "\\selectedData")
	for i in range(0, len(removeIndex)):
		shutil.rmtree(dirname + "\\selectedData\\test" + str(removeIndex[i]))
		LabelsIndex.remove(removeIndex[i])

	print("Updated Labels Index:", LabelsIndex)

	#update the numTests file
	numberTestFiles = open(dirname + "\\selectedData\\TestNumber.txt","w+")
	print("New File Size: ", len(LabelsIndex))
	numberTestFiles.write(str(len(LabelsIndex)))
	numberTestsSel = len(LabelsIndex)

	#update the numbering on remaining files
	for i in range(0, len(LabelsIndex)):
		os.rename((dirname + "\\selectedData\\test" + str(LabelsIndex[i])), (dirname + "\\selectedData\\test"+ str(i)))

	print("Data Stored in: ", dirname, "\\selectedData")

if __name__ == '__main__':
	main()