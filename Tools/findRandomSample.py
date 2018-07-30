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

Tool to create a random subsample of the larger dataset with an equal distribution of each exercise. Currently set up to
sample all 11 possible exercises.

'''

import random
import io
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

fileNumbers = random.sample(range(0, 986), 986)
totalNumber = 0
listOfExercises = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
finalFiles = []
finalLabels = []

print(fileNumbers)

i = 0
while (totalNumber < 110): 
	for line in open("C:\\Users\\Admin\\BlakeDeepak\\DataCollection\\test" + str(fileNumbers[i])+ "\\label.csv"):
		temporaryLabel = line.split()
		currentLabel = str(temporaryLabel[0])

	print (currentLabel)
	print (i , fileNumbers[i])

	if currentLabel.lower() == "y" and listOfExercises[0] < 10:
		listOfExercises[0] = listOfExercises[0] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "cat" and listOfExercises[1] < 10:
		listOfExercises[1] = listOfExercises[1] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "supine" and listOfExercises[2] < 10:
		listOfExercises[2] = listOfExercises[2] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "seated" and listOfExercises[3] < 10:
		listOfExercises[3] = listOfExercises[3] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "sumo" and listOfExercises[4] < 10:
		listOfExercises[4] = listOfExercises[4] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "mermaid" and listOfExercises[5] < 10:
		listOfExercises[5] = listOfExercises[5] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "towel" and listOfExercises[6] < 10:
		listOfExercises[6] = listOfExercises[6] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "trunk" and listOfExercises[7] < 10:
		listOfExercises[7] = listOfExercises[7] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "wall" and listOfExercises[8] < 10:
		listOfExercises[8] = listOfExercises[8] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "pretzel" and listOfExercises[9] < 10:
		listOfExercises[9] = listOfExercises[9] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "oov" and listOfExercises[10] < 10: #OOV
		listOfExercises[10] = listOfExercises[10] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	i = i + 1

print(len(finalFiles))
print (finalLabels)
print(listOfExercises)
print(totalNumber)

for i in range(0 , len(finalFiles)):
	shutil.copytree("C:\\Users\\Admin\\BlakeDeepak\\DataCollection\\test" + str(finalFiles[i]) , "C:\\Users\\Admin\\BlakeDeepak\\DataCollectionSample\\test" + str(i))
