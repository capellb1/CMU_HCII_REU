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

This code is a tool to count the number of each exercise in a dataset

'''

listOfExercises = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range (0, 110):
	for line in open("C:\\Users\\Admin\\BlakeDeepak\\DataCollectionSample\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			currentLabel = str(temporaryLabel[0])
			print(currentLabel)
			if currentLabel.lower() == "y" :
				listOfExercises[0] = listOfExercises[0] + 1

			elif currentLabel.lower() == "cat":
				listOfExercises[1] = listOfExercises[1] + 1
	
			elif currentLabel.lower() == "supine" :
				listOfExercises[2] = listOfExercises[2] + 1

			elif currentLabel.lower() == "seated" :
				listOfExercises[3] = listOfExercises[3] + 1
			
			elif currentLabel.lower() == "sumo":
				listOfExercises[4] = listOfExercises[4] + 1
				
			elif currentLabel.lower() == "mermaid" :
				listOfExercises[5] = listOfExercises[5] + 1
				

			elif currentLabel.lower() == "towel" :
				listOfExercises[6] = listOfExercises[6] + 1
				

			elif currentLabel.lower() == "trunk":
				listOfExercises[7] = listOfExercises[7] + 1
				
			elif currentLabel.lower() == "wall" :
				listOfExercises[8] = listOfExercises[8] + 1
				
			elif currentLabel.lower() == "pretzel":
				listOfExercises[9] = listOfExercises[9] + 1
				
			elif currentLabel.lower() == "oov" : #OOV
				listOfExercises[10] = listOfExercises[10] + 1
				
print (listOfExercises)