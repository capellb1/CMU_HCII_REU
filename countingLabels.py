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