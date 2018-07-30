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

A reload model binary variant to accompany the poise detector model variant. Performs same function as the standard relaod model,
however it also includes the calculations and probability classifier required to create an ROC curve
'''

#Import Libraries
import math
import io

#to get rid of warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Tensorflow and Data Processing Library
import tensorflow as tf
from tensorflow.python.data import Dataset
import numpy as np
import pandas as pd
import math 

#Display libraries for visualization
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
import seaborn as sns
import glob
import statistics as stat

#Define Flags to change the Hyperperameters and other variables
tf.app.flags.DEFINE_integer('batch_size',1000,'number of randomly sampled images from the training set')
tf.app.flags.DEFINE_float('learning_rate',0.001,'how quickly the model progresses along the loss curve during optimization')
tf.app.flags.DEFINE_integer('epochs',10,'number of passes over the training data')
tf.app.flags.DEFINE_float('regularization_rate',0.01,'Strength of regularization')
tf.app.flags.DEFINE_string('regularization', 'Default', 'This is the regularization function used in cost calcuations')
tf.app.flags.DEFINE_string('activation', 'Default', 'This is the activation function to use in the layers')
tf.app.flags.DEFINE_string('label', 'test1', 'This is the label name where the files are saved')
tf.app.flags.DEFINE_string('source', 'Position', 'What files to draw data frome (Task, Velocity, Position)')
tf.app.flags.DEFINE_string('arch', 'method1', 'This specifies the architecture used')
tf.app.flags.DEFINE_boolean('position', False, 'Determines if the position data is included when training')
tf.app.flags.DEFINE_boolean('velocity', False, 'Determines if the velocity data is included when training')
tf.app.flags.DEFINE_boolean('test', False, 'What mode are we running this model on. True runs the testing function')
tf.app.flags.DEFINE_boolean('verbose', False, 'Determines how much information is printed into the results file')
tf.app.flags.DEFINE_string('refinement', "None", 'Determines which refinement process to use')
tf.app.flags.DEFINE_integer('refinement_rate',0,'Determines the number of joints to include in the data')
tf.app.flags.DEFINE_boolean('task', False, 'Determines if the task data is included when training')
tf.app.flags.DEFINE_boolean('save', False, 'Determines wether the model is saved')

FLAGS = tf.app.flags.FLAGS

DATA_FOLDER = "selectedData"
#THRESHOLD = 0.30

batchIndex = 0

arch = FLAGS.arch
numberClasses = 2
if (arch == 'method1'):
	hiddenLayer1 = 60
	hiddenLayer2 = 60

elif (arch == 'method2'):
	hiddenLayer1 = 40
	hiddenLayer2 = 40
	hiddenLayer3 = 40

elif (arch == 'method3'):
	hiddenLayer1 = 30
	hiddenLayer2 = 30
	hiddenLayer3 = 30
	hiddenLayer4 = 30

else:
	hiddenLayer1 = 24
	hiddenLayer2 = 24
	hiddenLayer3 = 24
	hiddenLayer4 = 24
	hiddenLayer5 = 24

#list of all possible files
file_names_super =[
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

def writeFolderLabel():
	'''
		Creating a unuiqe folder name to save the results

		Returns
			String
	'''
	epochsLable = str(FLAGS.epochs)
	learning_rateLable = str(FLAGS.learning_rate)
	regularization_rateLable = str(FLAGS.regularization_rate)
	if(FLAGS.position):
		positionLable = "Position"
	else:
		positionLable = ""

	if(FLAGS.velocity):
		velocityLable = "Velocity"
	else:
		velocityLable = ""

	if(FLAGS.task):
		taskLable = "Task"
	else:
		taskLable = ""

	refinementLable = str(FLAGS.refinement)
	refinement_rateLable = str(FLAGS.refinement_rate)
	folderName = FLAGS.label + "E" + epochsLable + "LR" + learning_rateLable + FLAGS.activation + FLAGS.regularization + "RR" + regularization_rateLable  +  positionLable + velocityLable + taskLable + FLAGS.arch + "Ref" + refinementLable + "RefR" + refinement_rateLable

	return folderName

def calcNumTests():
	dirname = os.path.realpath('.')
	filename = dirname + '\\' + DATA_FOLDER + '\\TestNumber.txt'
	numberTestFiles = open(filename,"r")
	numberTests = numberTestFiles.read()
	if FLAGS.verbose:
		print("Number of Filed Detected: ", numberTests)
		resultsFile.write("Number of Filed Detected: " + str(numberTests) + '\n')

	return numberTests

def calcMaxEntries():
	maxEntries = 0
	timeScores = []
	for i in range(0,int(numberTests)):
		numEntries = 0
		for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+ "\\Position_" + file_names_super[0]):
			numEntries = numEntries + 1
		if numEntries > maxEntries:
			maxEntries = numEntries	
		timeScores.append(numEntries)
	
	if FLAGS.verbose:
		print("Maximum Number of Entries in a Single Exercise: ", maxEntries)
		resultsFile.write("Maximum Number of Entries in Single Exercise: " + str(maxEntries) + '\n')

	return maxEntries, timeScores

def calcBodySize():
	'''
		Establishes how many files will be read from, indepenent of the type of refinement

		Returns:
			int (number of joints used)
	'''
	if FLAGS.refinement_rate == 25:
		return 19

	elif FLAGS.refinement_rate == 50:
		return 13

	elif FLAGS.refinement_rate == 75:
		return 6

	else:
		return 25

def uniformRefinement():
	'''
		Applies uniform refinement. Changes the joints being used to train the data
		between predetermined levels. As refinement_rate increases, the number of joints
		decreases

		Returns:
			List of selected filenames
	'''
	if (FLAGS.refinement_rate == 0):
		file_names = file_names_super
		return file_names


	elif (FLAGS.refinement_rate == 25):
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
		'ElbowLeft.csv',     
		'WristLeft.csv',      
		'KneeRight.csv',    
		'AnkleRight.csv',   
		'FootRight.csv',     
		'KneeLeft.csv',
		'AnkleLeft.csv',     
		'FootLeft.csv']
		return file_names

	elif (FLAGS.refinement_rate == 50):
		file_names =[
		'Head.csv',          
		'ShoulderRight.csv', 
		'ShoulderLeft.csv',  
		'HipRight.csv',
		'HipLeft.csv', 
		'ElbowRight.csv',    
		'WristRight.csv',      
		'ElbowLeft.csv',     
		'WristLeft.csv',         
		'KneeRight.csv',    
		'AnkleRight.csv',       
		'KneeLeft.csv',
		'AnkleLeft.csv']
		return file_names

	elif (FLAGS.refinement_rate == 75):
		file_names =[         
		'ShoulderRight.csv', 
		'ShoulderLeft.csv',      
		'WristRight.csv',        
		'WristLeft.csv',            
		'AnkleRight.csv',       
		'AnkleLeft.csv']
		return file_names

def calcSections():
	'''
		Determines the number of datasets being used. Values range from 0-3.
		Used for matrix size allocation

		Returns:
			int numSections
	'''
	numSections = 0
	if FLAGS.position:
		numSections = numSections + 1
	if FLAGS.velocity:
		numSections = numSections + 1
	if FLAGS.task:
		numSections = numSections + 1	
	if numSections == 0:
		print("NO DATA SELECTED")

	if FLAGS.verbose:
		print("Number of sections: ", numSections)
		resultsFile.write("Number of datasets: " + str(numSections) + '\n')

	return numSections

def oneHot(labels):
	'''
		Accepts a list of labels and encodes each text label as a one hot encoding in an array with length = numClasses.
		Returns the list of encoded labels

		Accepts:
			list Labels

		Returns:
			list one_hot_labels
	'''
	one_hot_labels = []
	for i in range(0,len(labels)):
		if labels[i].lower() == "seated":
			one_hot_labels.append([1,0])
		elif labels[i].lower() == "towel":
			one_hot_labels.append([0,1])

	one_hot_labels = np.asarray(one_hot_labels)
	print("Lable Encoding Complete")
	return one_hot_labels

def findExercise (predictions):
	'''
	Reverses the encoding process to generate a list of strings.
	Used for increased readability

	Accepts:
		List of Integers ranging from 0 to numClasses

	Returns:
		List of Strings of the same length
	'''
	one_hot_labels = []
	for i in range(0,len(predictions)):
		if predictions[i] == 0:
			one_hot_labels.append("seated")
		elif predictions[i] == 1:
			one_hot_labels.append("towel")
	
	one_hot_labels = np.asarray(one_hot_labels)
	if FLAGS.verbose:
		print("Label Conversion Complete")
	return one_hot_labels

def tailor(i, refinement_rate):
	'''
		Scores each bodypart to reflect the amount of activity present in the joint. This
		information and a predetermined cutoff point (refinement_rate) is used to select
		which joints' data to use. Specialized to the person and exercise

		Accepts:
			int i (number of file examining, follows file name format)
			int refinement_rate

		Returns:
			List of Strings corresponding to the names of the most active joints
	'''

	jointActivity = []
	for j in range(0,24):
		activitySum = 0
		for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+ "\\Position_" + file_names_super[j]):
			row = line.split(',')
			for l in range(0,3):
				activitySum = activitySum + int(row[l])

		jointActivity.append((activitySum,j))

	jointActivity.sort()

	jointIndexActivity = [x[1] for x in jointActivity]

	if refinement_rate == 0:
		return uniformRefinement()
	
	elif refinement_rate == 25:
		selectedJoints = jointIndexActivity[-20:-1]
	
	elif refinement_rate == 50:
		selectedJoints = jointIndexActivity[-14:-1]
	
	elif refinement_rate == 75:
		selectedJoints = jointIndexActivity[-7:-1]

	new_file_names = []

	for x in selectedJoints:
		new_file_names.append(file_names_super[x])

	if FLAGS.verbose:
		print("New file names:", new_file_names)

	return new_file_names

def multilayer_perception(x, weights, biases):
	'''
		Define the activation layer and mathematical operations to occur at each level.
		Creates the model

		Accepts:
			dict('string':tf.varable) Weights, Biases
			x (input data of same size as Weights and Biases)

		Returns:
			outlayer (Structure of the model)

	'''
	activation = FLAGS.activation
	if (arch == "method1" and activation == "Sigmoid"):
		print('Activation Layer: sigmoid \nArchitecture Used: method2 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		outLayer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method1" and activation == "Tanh"):
		print('Activation Layer: tanh \nArchitecture Used: method2 \n')
		#Layers
		layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		outLayer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method1" and activation == "Relu"):
		print('Activation Layer: relu \nArchitecture Used: method2 \n')
		#Layers
		layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		outLayer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method1" and activation == "Default"):
		print('Activation Layer: none \nArchitecture Used: method2 \n')
		#Layers
		layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
		outLayer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method2" and activation == "Sigmoid"):
		print('Activation Layer: sigmoid \nArchitecture Used: method1 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method2" and activation == "Tanh"):
		print('Activation Layer: tanh \nArchitecture Used: method1 \n ')
		#Layers
		layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.tanh(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method2" and activation == "Relu"):
		print('Activation Layer: relu \nArchitecture Used: method1 \n ')
		#Layers
		layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method2" and activation == "Default"):
		print('Activation Layer: none \nArchitecture Used: method1 \n ')
		#Layers
		layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
		layer3 = tf.add(tf.matmul(layer2, weights['h3']), biases['b3'])
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	
	elif (arch == "method3" and activation == "Sigmoid"):
		print('Activation Layer: sigmoid \nArchitecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method3" and activation == "Tanh"):
		print('Activation Layer: tanh \nArchitecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method3" and activation == "Relu"):
		print('Activation Layer: relu \nArchitecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method3" and activation == "Default"):
		print('Activation Layer: none \nArchitecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer
	
	elif (arch == "method4" and activation == "Sigmoid"):
		print('Activation Layer: sigmoid \nArchitecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		layer5 = tf.nn.sigmoid(tf.add(tf.matmul(layer4, weights['h5']), biases['b5']))
		outLayer = tf.add(tf.matmul(layer5, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method4" and activation == "Tanh"):
		print('Activation Layer: tanh \nArchitecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		layer5 = tf.nn.sigmoid(tf.add(tf.matmul(layer4, weights['h5']), biases['b5']))
		outLayer = tf.add(tf.matmul(layer5, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method4" and activation == "Relu"):
		print('Activation Layer: relu \nArchitecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		layer5 = tf.nn.sigmoid(tf.add(tf.matmul(layer4, weights['h5']), biases['b5']))
		outLayer = tf.add(tf.matmul(layer5, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method4" and activation == "Default"):
		print('Activation Layer: none \nArchitecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		layer5 = tf.nn.sigmoid(tf.add(tf.matmul(layer4, weights['h5']), biases['b5']))
		outLayer = tf.add(tf.matmul(layer5, weights['out']), biases['out'])
		return outLayer

def nextBatch(batchSize, trainNumber):
	'''
		Determines which portion of the data to feed into the net

		Accepts:
			int batchSize
			int trainNumber (number of files in the dataset)

		returns:
			int startIndex
			int endIndex
	'''
	global batchIndex
	start = batchIndex
	batchIndex += batchSize
	if batchIndex > trainNumber:
		batchIndex = trainNumber
	end = batchIndex
	return start, end

def extractData():
	'''
		Moves data from the text files into flattened arrays.
		Each time stamp is a single row and has a corresponding event label
			[Arm1xyz, Head1xyz, Foot1xyz, ...] EVENT 1
			[Arm2xyz, Head2xyz, Foot2xyz, ...] EVENT 2
		
		Parameters: None
		Returns:
			nparray shuffledlabels
			nparray shuffledData
	'''
	#average
	timeScores2 = []
	for i in range(0, len(timeScores)):
		timeScores2.append(timeScores[i]//2)

	data =  np.empty((sum(timeScores2), int(bodySize*3*numSections)))
	sample = True

	numTimeStamps = 0
	labels = []
	edges = []
	c=0
	for i in range(0, int(numberTests)):
		#Determine the number of time stamps in this event
		w=0

		if FLAGS.refinement == "Tailored":
			global file_names 
			file_names = tailor(i, FLAGS.refinement_rate)

		for l in range(numTimeStamps,numTimeStamps+(timeScores[i]//2)):
			k=0
			for j in range(0, bodySize):
				if FLAGS.position:

					fp = open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+ "\\Position_" + file_names[j])
					for n, line in enumerate(fp):
						if n == w:
							row = line.split(',')
							for m in range(0,3):
								data[l][k]= row[m]
								k = k + 1
			
				if FLAGS.velocity:
					fp = open(dirname + "\\"+ DATA_FOLDER +"\\test"+ str(i)+ "\\Velocity_" + file_names[j])

					for n, line in enumerate(fp):
						if n == w:
							row = line.split(',')
							for m in range(0,3):
								data[l][k]= row[m]
								k = k + 1
				if FLAGS.task:
					fp = open(dirname + "\\"+ DATA_FOLDER +"\\test"+ str(i)+ "\\Task_" + file_names[j])

					for n, line in enumerate(fp):
						if n == w:
							row = line.split(',')
							for m in range(0,3):
								data[l][k]= row[m]
								k = k + 1
				
			for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+ "\\label.csv"):
				temporaryLabel = line.split()
				labels.append(str(temporaryLabel[0]))

			w=w+2	
		
		edges.append(numTimeStamps)
		
		numTimeStamps = (timeScores[i]//2) + numTimeStamps
	
	fp.close()
	#shuffle the data
	shuffledData = np.empty(data.shape, dtype=data.dtype)
	shuffledLabels = labels
	permutation = np.random.RandomState(seed=42).permutation(len(labels))
	for old_index, new_index in enumerate(permutation):
		shuffledData[new_index] = data[old_index]
		shuffledLabels[new_index] = labels[old_index]

	shuffledLabels = np.asarray(shuffledLabels)
	return shuffledData, shuffledLabels

def calcLableDist():
	y_count = 0.1
	seated_count = 0.1
	sumo_count = 0.1
	mermaid_count = 0.1
	towel_count = 0.1
	wall_count = 0.1
	oov_count = 0.1

	labelDist = []
	labels = []

			#seperate the label from the name and event number stored within the label.csv file(s)
	for i in range(0, int(numberTests)):
		for line in open(dirname + "\\"+ DATA_FOLDER +"\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			labels.append(str(temporaryLabel[0]))

	for i in range(0, len(labels)):

		if (labels[i].lower() == "y"):
			y_count = y_count + 1
		elif (labels[i].lower() == "seated"):
			seated_count = seated_count + 1
		elif (labels[i].lower() == "sumo"):
			sumo_count = sumo_count + 1
		elif (labels[i].lower() == "mermaid"):
			mermaid_count = mermaid_count + 1
		elif (labels[i].lower() == "towel"):
			towel_count = towel_count + 1
		elif (labels[i].lower() == "wall"):
			wall_count = wall_count + 1
		else:
			oov_count = oov_count + 1

	labelDist.append(y_count)
	labelDist.append(seated_count)
	labelDist.append(sumo_count)
	labelDist.append(mermaid_count)
	labelDist.append(towel_count)
	labelDist.append(wall_count)
	labelDist.append(oov_count)

	return labelDist 

def std(data, numberTests):
	dataByBody = []
	means = []
	stdevs = []

	mean = 0
	stdev = 0
	for k in range(0,bodySize*3*numSections):
		bodypartData = []
		for l in range(0,len(data)):
			bodypartData.append(data[l][k])

		mean = stat.mean(bodypartData)
		stdev = stat.stdev(bodypartData)
		dataByBody.append(bodypartData)
		means.append(mean)
		stdevs.append(stdev)

		for j in range(0, len(bodypartData)):
			dataByBody[k][j] = (dataByBody[k][j] - mean)/stdev

	for l in range(0,len(data)):
		for k in range(0,bodySize*3*numSections):
			data[l][k] = dataByBody[k][l]

	return data, means, stdevs

def stdTest(data, numberTests, mean, stdev):
	dataByBody = []
	for k in range(0,bodySize*3*numSections):
		bodypartData = []
		for l in range(0,len(data[:])):
			bodypartData.append(data[l][k])

		dataByBody.append(bodypartData)

		for j in range(0, len(bodypartData)):
			dataByBody[k][j] = (dataByBody[k][j] - mean[k])/stdev[k]

	for l in range(0,len(data[:])):
		for k in range(0,bodySize*3*numSections):
			data[l][k] = dataByBody[k][l]

	return data

def draw(predictions, correctPredictions, dataTask, predicList):	
	'''
		Creates graphs that plot the accuracy of predictions for every frame in an action (Blue). Below, on 
		the same image, a graph representing the number of tasks detected across each body part for 
		that each frame (Red). This occurs for each example in the data.
		
		The function will also output a cumulative histogram for each action with more
		than one example (Green). This cumulative histogram overlays the accuracy of every example of a given
		exercise.
	'''
	start = 0	
	dataRecord = np.zeros((7,maxEntries))
	probRecord = np.zeros((7,maxEntries))
	taskRecord = np.zeros((7,maxEntries))
	offset = 0
	#range from 0-10
	timeRecord = [[0],[0],[0],[0],[0],[0],[0]]
	
	print("Predictions to get Graphed:", correctPredictions)		
	for i in range (0, int(numberTests)):
		graphData = []
		graphDataX = []
		taskData = []
		for j in range (0, timeScores[i]):
			if predictions[start+j] == correctPredictions[start]:
				graphData.append(1)
						
				if (correctPredictions[start] == "y") and (predicList[j][0] > THRESHOLD):
					dataRecord[0][j] = dataRecord[0][j] + 1
				elif (correctPredictions[start] == "seated"):
					dataRecord[1][j] = dataRecord[1][j] + 1
				elif (correctPredictions[start] == "sumo"):
					dataRecord[2][j] = dataRecord[2][j] + 1
				elif (correctPredictions[start] == "mermaid"):
					dataRecord[3][j] = dataRecord[3][j] + 1
				elif (correctPredictions[start] == "towel"):
					dataRecord[4][j] = dataRecord[4][j] + 1
				elif (correctPredictions[start] == "wall"):
					dataRecord[5][j] = dataRecord[5][j] + 1
				elif (correctPredictions[start] == "oov"):
					dataRecord[6][j] = dataRecord[6][j] + 1

			else:
				graphData.append(0)
				
				
			graphDataX.append(j)

			#calculate probability rather than raw numbers
			for k in range(0,7):
				probRecord[k][j] = dataRecord[k][j]/labelDist[k]
			


		offset = timeScores[i]
				
		if (correctPredictions[start] == "y"):
			timeRecord[0].append(graphDataX[-1])
		elif (correctPredictions[start] == "seated"):
			timeRecord[1].append(graphDataX[-1])
		elif (correctPredictions[start] == "sumo"):
			timeRecord[2].append(graphDataX[-1])
		elif (correctPredictions[start] == "mermaid"):
			timeRecord[3].append(graphDataX[-1])
		elif (correctPredictions[start] == "towel"):
			timeRecord[4].append(graphDataX[-1])
		elif (correctPredictions[start] == "wall"):
			timeRecord[5].append(graphDataX[-1])
		elif (correctPredictions[start] == "oov"):
			timeRecord[6].append(graphDataX[-1])
				
		if FLAGS.verbose:
			print(timeScores[i])
			print("My preditions", predictions[start:start + timeScores[i]])
			print ("Correct predictions", correctPredictions[start])
			print("Length of single test time scores: ", graphDataX[-1])
			print(graphData)

		width = .99
		plt.bar(graphDataX, graphData, width, facecolor='blue')
		plt.xlabel("Frames")
		plt.annotate(str(graphDataX[-1]), xy = (graphDataX[-1], 1))
		plt.ylabel("Accuracy (Bool)")
		plt.title("Test" + str(i) +": " + str(correctPredictions[start]))
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(newDir +"\\test" + str(i) + str(correctPredictions[start]) + ".png")
		plt.close()
		start = start + timeScores[i]
		
	totalDataX = range(0, maxEntries)
			
	
	averageTs = []
	for i in range(0,7):
		current_avg = sum(timeRecord[i])/float(len(timeRecord[i]))
		averageTs.append(int(current_avg))
	

	width = .99

	#y
	if len(timeRecord[0]) > 1:
		plt.bar(totalDataX, probRecord[0][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Probability of an Accurate Prediction")
		plt.title("Cumulative Y Exercise Data")
		annotation = "Average Timespan: " + str(averageTs[0])
		plt.annotate(annotation, xy = (averageTs[0], 0.5))
		plt.grid(True)
		
		plt.savefig(newDir +"\\yTotalData.png")
		plt.close()

	#Seated
	if len(timeRecord[1]) > 1:
		plt.bar(totalDataX, probRecord[1][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Probability of an Accurate Prediction")
		plt.title("Cumulative Seated Exercise Data")
		annotation = "Average Timespan: " + str(averageTs[1])
		plt.annotate(annotation, xy = (averageTs[1], 0.5))
		plt.grid(True)
		

		plt.savefig(newDir +"\\seatedTotalData.png")
		plt.close()

	#sumo
	if len(timeRecord[2]) > 1:
		plt.bar(totalDataX, probRecord[2][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Probability of an Accurate Prediction")
		plt.title("Cumulative Sumo Exercise Data")
		annotation = "Average Timespan: " + str(averageTs[2])
		plt.annotate(annotation, xy = (averageTs[2], 0.5))
		plt.grid(True)
		
		plt.savefig(newDir +"\\sumoTotalData.png")
		plt.close()

	#mermaid
	if len(timeRecord[3]) > 1:
		plt.bar(totalDataX, probRecord[3][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Probability of an Accurate Prediction")
		plt.title("Cumulative Mermaid Exercise Data")
		annotation = "Average Timespan: " + str(averageTs[3])
		plt.annotate(annotation, xy = (averageTs[3], 0.5))
		plt.grid(True)
		
		plt.savefig(newDir +"\\mermaidTotalData.png")
		plt.close()

	#towel
	if len(timeRecord[4]) > 1:
		plt.bar(totalDataX, probRecord[4][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Probability of an Accurate Prediction")
		plt.title("Cumulative Towel Exercise Data")
		annotation = "Average Timespan: " + str(averageTs[4])
		plt.annotate(annotation, xy = (averageTs[4], 0.5))
		plt.grid(True)
		

		plt.savefig(newDir +"\\towelTotalData.png")
		plt.close()

	#wall
	if len(timeRecord[5]) > 1:
		plt.bar(totalDataX, probRecord[5][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Probability of an Accurate Prediction")
		plt.title("Cumulative Wall Exercise Data")
		annotation = "Average Timespan: " + str(averageTs[5])
		plt.annotate(annotation, xy = (averageTs[5], 0.5))
		plt.grid(True)
		
		plt.savefig(newDir +"\\wallTotalData.png")
		plt.close()

	#oov
	if len(timeRecord[6]) > 1:
		plt.bar(totalDataX, probRecord[6][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Probability of an Accurate Prediction")
		plt.title("Cumulative OOV Data")
		annotation = "Average Timespan: " + str(averageTs[6])
		plt.annotate(annotation, xy = (averageTs[6], 0.5))
		plt.grid(True)

		plt.savefig(newDir +"\\oovTotalData.png")
		plt.close()


if FLAGS.refinement == "Uniform":
	file_names = uniformRefinement()

elif FLAGS.refinement == "None":
	file_names = file_names_super

dirname = os.path.realpath('.')

folderName = writeFolderLabel()

#Open file used to store accuracy scores and any other printed data
newDir = dirname + '\\Models&Results\\' + folderName
if not (os.path.exists(newDir)):
	os.makedirs(newDir)

resultsFile = open(newDir + '\\Results.txt',"w+")

numSections = calcSections()

bodySize = calcBodySize()

numberTests = calcNumTests()

maxEntries, timeScores = calcMaxEntries()

labelDist = calcLableDist()
print("Label Distribution: ", labelDist)

def main(argv = None):
	'''
		Call all methods defined above and determine the shape of the network. This
		portion also defines and stores all of the weights and biases. Defines optimizer and trains
		the network
	'''
	learningRate = FLAGS.learning_rate
	epochsTrained = FLAGS.epochs
	batchSize = FLAGS.batch_size
	#display step

	data, labels = extractData()
	labelText = labels
	labels = oneHot(labels)

	data, means, stdevs = std(data, len(labels))

	inputLayer = bodySize*3*numSections

	#tf Graph input
	X = tf.placeholder(data.dtype, [None, inputLayer])
	Y = tf.placeholder(labels.dtype, [None, numberClasses])

	if (arch == 'method1'):
		weights = {
		'h1' : tf.Variable(tf.random_normal([inputLayer, hiddenLayer1], dtype=data.dtype, name='h1')),
		'h2' : tf.Variable(tf.random_normal([hiddenLayer1, hiddenLayer2], dtype=data.dtype, name='h2')),
		'out' : tf.Variable(tf.random_normal([hiddenLayer2, numberClasses], dtype=data.dtype, name='out'))
		}

		biases = {
		'b1' : tf.Variable(tf.random_normal([hiddenLayer1], dtype=data.dtype, name = 'b1')),
		'b2' : tf.Variable(tf.random_normal([hiddenLayer2], dtype=data.dtype, name = 'b2')),
		'out' : tf.Variable(tf.random_normal([numberClasses], dtype=data.dtype, name = 'outb'))
		}
	elif (arch == "method2"):
		weights = {
		'h1' : tf.Variable(tf.random_normal([inputLayer, hiddenLayer1], dtype=data.dtype, name='h1')),
		'h2' : tf.Variable(tf.random_normal([hiddenLayer1, hiddenLayer2], dtype=data.dtype, name ='h2')),
		'h3' : tf.Variable(tf.random_normal([hiddenLayer2, hiddenLayer3], dtype=data.dtype, name ='h3')),
		'out' : tf.Variable(tf.random_normal([hiddenLayer3, numberClasses], dtype=data.dtype, name = 'out'))
		}

		biases = {
		'b1' : tf.Variable(tf.random_normal([hiddenLayer1], dtype=data.dtype, name = 'b1')),
		'b2' : tf.Variable(tf.random_normal([hiddenLayer2], dtype=data.dtype, name = 'b2')),
		'b3' : tf.Variable(tf.random_normal([hiddenLayer3], dtype=data.dtype, name = 'b3')),
		'out' : tf.Variable(tf.random_normal([numberClasses], dtype=data.dtype, name = 'outb'))
		}
	elif (arch == "method3"):
		weights = {
		'h1' : tf.Variable(tf.random_normal([inputLayer, hiddenLayer1], dtype=data.dtype, name='h1')),
		'h2' : tf.Variable(tf.random_normal([hiddenLayer1, hiddenLayer2], dtype=data.dtype, name='h2')),
		'h3' : tf.Variable(tf.random_normal([hiddenLayer2, hiddenLayer3], dtype=data.dtype, name='h3')),
		'h4' : tf.Variable(tf.random_normal([hiddenLayer3, hiddenLayer4], dtype=data.dtype, name='h4')),
		'out' : tf.Variable(tf.random_normal([hiddenLayer4, numberClasses], dtype=data.dtype, name='out'))
		}

		biases = {
		'b1' : tf.Variable(tf.random_normal([hiddenLayer1], dtype=data.dtype, name = 'b1')),
		'b2' : tf.Variable(tf.random_normal([hiddenLayer2], dtype=data.dtype, name = 'b2')),
		'b3' : tf.Variable(tf.random_normal([hiddenLayer3], dtype=data.dtype, name = 'b3')),
		'b4' : tf.Variable(tf.random_normal([hiddenLayer4], dtype=data.dtype, name = 'b4')),		
		'out' : tf.Variable(tf.random_normal([numberClasses], dtype=data.dtype, name = 'bout'))
		}
	else:
		weights = {
		'h1' : tf.Variable(tf.random_normal([inputLayer, hiddenLayer1], dtype=data.dtype, name='h1')),
		'h2' : tf.Variable(tf.random_normal([hiddenLayer1, hiddenLayer2], dtype=data.dtype, name='h2')),
		'h3' : tf.Variable(tf.random_normal([hiddenLayer2, hiddenLayer3], dtype=data.dtype, name='h3')),
		'h4' : tf.Variable(tf.random_normal([hiddenLayer3, hiddenLayer4], dtype=data.dtype, name='h4')),
		'h5' : tf.Variable(tf.random_normal([hiddenLayer4, hiddenLayer5], dtype=data.dtype, name='h5')),
		'out' : tf.Variable(tf.random_normal([hiddenLayer5, numberClasses], dtype=data.dtype, name='out'))
		}

		biases = {
		'b1' : tf.Variable(tf.random_normal([hiddenLayer1], dtype=data.dtype, name = 'b1')),
		'b2' : tf.Variable(tf.random_normal([hiddenLayer2], dtype=data.dtype, name = 'b2')),
		'b3' : tf.Variable(tf.random_normal([hiddenLayer3], dtype=data.dtype, name = 'b3')),
		'b4' : tf.Variable(tf.random_normal([hiddenLayer4], dtype=data.dtype, name = 'b4')),
		'b5' : tf.Variable(tf.random_normal([hiddenLayer5], dtype=data.dtype, name = 'b5')),		
		'out' : tf.Variable(tf.random_normal([numberClasses], dtype=data.dtype, name = 'bout'))
		}

	saver = tf.train.Saver()


	modelPath =  newDir + "\\ExercisePredicter"
	with tf.Session() as sess:

		saver.restore(sess, modelPath )

		sess.run(weights)
		sess.run(biases)
		
		logits = multilayer_perception(data, sess.run(weights), sess.run(biases))
		pred = tf.nn.softmax(logits)

		probs = tf.reduce_max(pred, 1)
		probsIndex = tf.argmax(pred, 1)
		labelsIndex = tf.argmax(Y,1)
		probIndexRes = probsIndex.eval({X: data})
		labelIndexRes = labelsIndex.eval({Y: labels})
		probabilityResults = probs.eval({X: data})
		
		#Calculate ROC
		truePosList = []
		falsePosList = []

		for THRESHOLD in np.linspace(0,0.9,10):
			numTruePos = 0.0
			numFalseNeg = 0.0
			numFalsePos = 0.0
			numTrueNeg = 0.0

			passed = 0.0
			chosen = 0.0

			predicList = []

			for i in range(0, len(probabilityResults)):
				if  (probabilityResults[i] < THRESHOLD):
					passed = passed + 1
					#prediction rejected due to low confidence
					predicList.append((-1, 1, labelIndexRes[i]))
				else:
					#above confidence, will always accept a sick diagnosiss
					chosen = chosen + 1
					predicList.append((probabilityResults[i], 0, labelIndexRes[i]))

			print("")
			print(THRESHOLD, " , Omitted:", passed, " Allowed:", chosen)

			#For sake of ROC assume 0 = sick
			for j in range(0, len(predicList)):
			
				if (predicList[j][1] == 0) and (predicList[j][2] == 0):
					#predict sick is sick
					numTruePos = numTruePos + 1.0
					
				elif (predicList[j][1] == 0) and (predicList[j][2] == 1):
					#predict sick but healthy
					numFalsePos = numFalsePos + 1.0

				elif (predicList[j][1] == 1) and (predicList[j][2] == 1):
					#predict healthy is healthy
					numTrueNeg = numTrueNeg + 1.0

				elif (predicList[j][1] == 1) and (predicList[j][2] == 0):
					#predict healthy is sick
					numFalseNeg = numFalseNeg + 1.0



			print("TP", numTruePos)
			print("FP", numFalsePos)
			print("TN", numTrueNeg)
			print("FN", numFalseNeg)
			
			truePos = (numTruePos/(numTruePos+numFalseNeg))
			falsePos = (numFalsePos/(numFalsePos+numTrueNeg))
			print("True Positive Rate: ", truePos)
			truePosList.append(truePos)
			print("False Positive Rate: ", falsePos)
			falsePosList.append(falsePos)

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.plot(falsePosList,truePosList, label = 'Model')
		ax1.plot(np.linspace(0,1,11), np.linspace(0,1,11), label = 'Chance')
		plt.legend()
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.title("Receiver Operator Characteristic (ROC) Curve")
		plt.grid(True)
		plt.savefig(newDir +"\\ROC.png")
		plt.show()
		plt.close()

		

		if (FLAGS.test):
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print("Final Accuracy on new Data is: ", "{0:.2%}".format(sess.run(accuracy)), '\n')
			predictions = tf.argmax(pred,1).eval()
			predictions = findExercise(predictions) 
			print("My Predictions", predictions, '\n')
			print("Actual Labels", labelText)
		else: 		
			predictionsOg = tf.argmax(pred,1).eval()
			predictions = findExercise(predictionsOg) 
			correctPredictionsOg = tf.argmax(labels,1).eval()
			correctPredictions = findExercise(correctPredictionsOg)
			#draw(predictions, correctPredictions, dataTask, predicList)

		confusion_matrix = tf.confusion_matrix(correctPredictionsOg, predictionsOg).eval()
		print("Confusion Matrix: ")
		print(confusion_matrix)
		resultsFile.write("Confusion Matrix:" + str(confusion_matrix))


	
#needed in order to call main
if __name__ == '__main__':
	main()