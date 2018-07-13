import os

#determine num of files in dataset
dirname = os.path.realpath('.')
filename = dirname + '\\DataCollectionSample\\TestNumber.txt'
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()

labels = []

#Collect all the labels
for i in range(0, int(numberTests)):
	for line in open(dirname + "\\DataCollectionSample\\test" + str(i)+ "\\label.csv"):
		temporaryLabel = line.split()
		labels.append((str(temporaryLabel[0]),i))

#parse list for desired examples
removeIndex = [label[1] for label in labels if (label[0] == "Cat" or label[0] == "Trunk" or label[0] == "Supine") ]
LabelsIndex = [label[1] for label in labels]
print("Remove Index:", removeIndex)
print("Labels Index:", LabelsIndex)

#move desired exercises out of the folder
for i in range(0, len(removeIndex)):
	os.rename((dirname + "\\DataCollectionSample\\test" + str(removeIndex[i])), (dirname + "\\DiscardedData\\test"+ str(removeIndex[i])))
	LabelsIndex.remove(removeIndex[i])

print("Updated Labels Index:", LabelsIndex)

#update the numTests file
numberTestFiles = open(filename,"w")
print("New File Size: ", len(LabelsIndex))
numberTestFiles.write(str(len(LabelsIndex)))



#update the numbering on remaining files
for i in range(0, len(LabelsIndex)):
	os.rename((dirname + "\\DataCollectionSample\\test" + str(LabelsIndex[i])), (dirname + "\\DataCollectionSample\\test"+ str(i)))
