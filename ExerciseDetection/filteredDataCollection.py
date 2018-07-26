import io
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



dirname = os.path.realpath('.')
filename = dirname + '\\selectedData\\TestNumber.txt'
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()

print(dirname)
newDir = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow"
if not (os.path.exists(newDir)):
	os.makedirs(newDir)

#set up with start, end

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
	newDir2 = "C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow\\test" + str(i)
	if not (os.path.exists(newDir2)):
		os.makedirs(newDir2)

	for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\label.csv"):
		temporaryLabel = line.split()
		temporaryLabel = temporaryLabel[0]
	
	resultsFileL = open("C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow\\test" + str(i) +"\\label.csv", "a+")
	resultsFileL.write(temporaryLabel) 

	exerciseNumber = 0
	if temporaryLabel.lower() == "y":
		exerciseNumber = 0
	elif temporaryLabel.lower() == "seated":
		exerciseNumber = 1
	elif temporaryLabel.lower() == "sumo":
		exerciseNumber = 2
	elif temporaryLabel.lower() == "mermaid":
		exerciseNumber = 3
	elif temporaryLabel.lower() == "towel":
		exerciseNumber = 4
	elif temporaryLabel.lower() == "wall":
		exerciseNumber = 5
	else:
		exerciseNumber = 6

	print (exerciseNumber)
	print (windowTime[exerciseNumber*2])									
	k = 0
	for j in range(0,bodySize):
		resultsFileP = open("C:\\Users\\Admin\\BlakeDeepak\\CMU_HCII_REU\\ExerciseDetection\\DataWindow\\test" + str(i) + "\\Position_" + file_names[j], "a+")
		m = 0
		sample = True
		for line in open(dirname + "\\selectedData\\test" + str(i)+ "\\Position_" + file_names[j]):
			if m >= 2*windowTime[2*exerciseNumber] and m < 2*windowTime[2*exerciseNumber+ 1]:
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
			if m >= 2*windowTime[2*exerciseNumber] and m < 2*windowTime[2*exerciseNumber + 1]:
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
			if m >= 2*windowTime[2*exerciseNumber] and m < 2*windowTime[2*exerciseNumber + 1]:
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

