import io
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dirname = os.path.realpath('.')
filename = dirname + '\\Data\\TestNumber.txt'
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()

newDir = "C:\\Users\\Deepak Subramanian\\Documents\\Internship\\HCII Research (2018)\\CMU_HCII_REU\\DataWindow"
if not (os.path.exists(newDir)):
	os.makedirs(newDir)

#set up with start, end
windowTime = [50 ,100,50 ,100,50 ,100,50 ,100,50 ,100,50 ,100,50 ,100 ,50 ,100 ,50 ,100 ,50 ,100 ,50 ,100]

#list of all possible files
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

bodySize = 25
for i in range(0, int(numberTests)):
	newDir2 = "C:\\Users\\Deepak Subramanian\\Documents\\Internship\\HCII Research (2018)\\CMU_HCII_REU\\DataWindow\\test" + str(i)
	if not (os.path.exists(newDir2)):
		os.makedirs(newDir2)

	for line in open(dirname + "\\Data\\test" + str(i)+ "\\label.csv"):
		temporaryLabel = line.split()
		temporaryLabel = temporaryLabel[0]

	exerciseNumber = 0
	if temporaryLabel.lower() == "y":
		exerciseNumber = 0
	elif temporaryLabel.lower() == "cat":
		exerciseNumber = 1
	elif temporaryLabel.lower() == "supine":
		exerciseNumber = 2
	elif temporaryLabel.lower() == "seated":
		exerciseNumber = 3
	elif temporaryLabel.lower() == "sumo":
		exerciseNumber = 4
	elif temporaryLabel.lower() == "mermaid":
		exerciseNumber = 5
	elif temporaryLabel.lower() == "towel":
		exerciseNumber = 6
	elif temporaryLabel.lower() == "trunk":
		exerciseNumber = 7
	elif temporaryLabel.lower() == "wall":
		exerciseNumber = 8
	elif temporaryLabel.lower() == "pretzel":
		exerciseNumber = 9
	else:
		exerciseNumber = 10

	print (exerciseNumber)
	print (windowTime[exerciseNumber*2])									
	k = 0
	for j in range(0,bodySize):
		resultsFileP = open("C:\\Users\\Deepak Subramanian\\Documents\\Internship\\HCII Research (2018)\\CMU_HCII_REU\\DataWindow\\test" + str(i) + "\\Position_" + file_names[j], "a+")
		m = 0
		sample = False
		for line in open(dirname + "\\Data\\test" + str(i)+ "\\Position_" + file_names[j]):
			if m >= windowTime[2*exerciseNumber] and m < windowTime[2*exerciseNumber+ 1]:
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
		resultsFileV = open("C:\\Users\\Deepak Subramanian\\Documents\\Internship\\HCII Research (2018)\\CMU_HCII_REU\\DataWindow\\test" + str(i) + "\\Velocity_" + file_names[j], "a+")
		m = 0
		sample = False
		for line in open(dirname + "\\Data\\test" + str(i)+ "\\Velocity_" + file_names[j]):
			if m >= windowTime[2*exerciseNumber] and m < windowTime[2*exerciseNumber + 1]:
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

		resultsFileT = open("C:\\Users\\Deepak Subramanian\\Documents\\Internship\\HCII Research (2018)\\CMU_HCII_REU\\DataWindow\\test" + str(i) + "\\Task_" + file_names[j], "a+")
		m = 0
		sample = False
		for line in open(dirname + "\\Data\\test" + str(i)+ "\\Task_" + file_names[j]):
			if m >= windowTime[2*exerciseNumber] and m < windowTime[2*exerciseNumber + 1]:
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

