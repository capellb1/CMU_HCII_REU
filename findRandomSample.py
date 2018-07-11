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
while (totalNumber < 100): 
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
		listOfExercises[0] = listOfExercises[1] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "supine" and listOfExercises[2] < 10:
		listOfExercises[0] = listOfExercises[2] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "seated" and listOfExercises[3] < 10:
		listOfExercises[0] = listOfExercises[3] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "sumo" and listOfExercises[4] < 10:
		listOfExercises[0] = listOfExercises[4] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "mermaid" and listOfExercises[5] < 10:
		listOfExercises[0] = listOfExercises[5] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "towel" and listOfExercises[6] < 10:
		listOfExercises[0] = listOfExercises[6] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "trunk" and listOfExercises[7] < 10:
		listOfExercises[0] = listOfExercises[7] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "wall" and listOfExercises[8] < 10:
		listOfExercises[0] = listOfExercises[8] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "pretzel" and listOfExercises[9] < 10:
		listOfExercises[0] = listOfExercises[9] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	elif currentLabel.lower() == "oov" and listOfExercises[10] < 10: #OOV
		listOfExercises[0] = listOfExercises[10] + 1
		totalNumber = totalNumber + 1
		finalFiles.append(fileNumbers[i])
		finalLabels.append(currentLabel)

	i = i + 1

print(finalFiles)
print (finalLabels)

#os.makedirs("C:\\Users\\Admin\\BlakeDeepak\\DataCollectionSample\\test0")
for i in range(0 , len(finalFiles)):
	shutil.copytree("C:\\Users\\Admin\\BlakeDeepak\\DataCollection\\test" + str(finalFiles[i]) , "C:\\Users\\Admin\\BlakeDeepak\\DataCollectionSample\\test" + str(i))
