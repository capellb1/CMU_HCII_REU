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

Tool used to change the labels from an integer value to their appropraite string label
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

for i in range (0, 986):
	f2 = open("C:\\Users\\Admin\\BlakeDeepak\\DataCollection\\test" + str(i)+ "\\label.csv", 'r')
	currentLabel = (f2.read())
	f2.close()

	for line in open("C:\\Users\\Admin\\BlakeDeepak\\DataCollection\\test" + str(i)+ "\\label.csv"):
		temporaryLabel = line.split()
	print (i, temporaryLabel)

	if (currentLabel[1] != '\r'):
		currentLabel = currentLabel[0] + currentLabel[1]
	else:
		currentLabel = currentLabel[0]
	f = open("C:\\Users\\Admin\\BlakeDeepak\\DataCollection\\test" + str(i)+ "\\label.csv", 'w')

	if currentLabel == "1":
		f.write("Y")
	elif currentLabel == "2":
		f.write("Cat")

	elif currentLabel == "3":
		f.write("Supine")

	elif currentLabel == "4":
		f.write("Seated")

	elif currentLabel == "5":
		f.write("Sumo")

	elif currentLabel == "6":
		f.write("Mermaid")

	elif currentLabel == "7":
		f.write("Towel")

	elif currentLabel == "8":
		f.write("Trunk")

	elif currentLabel == "9":
		f.write("Wall")

	elif currentLabel == "10":
		f.write("Pretzel")

	elif currentLabel == "11":
		f.write("OOV")

	#else: 
		f.write(currentLabel)

	f.close()

