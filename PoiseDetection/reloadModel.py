'''
CMU HCII REU Summer 2018
PI: Dr. Sieworek
Students:  Blake Capella & Deepak Subramanian
Date: 07/09/18

The following code is used to either train or visualize the results of a neural net. The project's goal is to analyze the
difference in performance between multi frame analysis (exercise_detector) and single frame analysis (poise_detector). Built
in to each of the files are a large number of flags used change numerous features ranging from input data
to network architecture and other hyperparameters. For more detailed information on the flags, see the code or visit 
https://github.com/capellb1/CMU_HCII_REU.git

Many flags might not be used in this file, they were included for consistency between the multiple training files.
	poise_detector_mk*.py
	exercise_detection_mk*.py
	reloadModel.py

	Unless otherwise stated, assume highest number to be the most current/advanced file

MUST HAVE AT LEAST 5 files in order to be used

Assumes that you are reading from a data library constructed by the task_sequencer_v2.pde file
If not, organize your data as follows:
	Data
		test0
			Position_Head.csv (organized by x,y,z,ts)
			Position_Neck.csv
			.
			.
			.
			Velocity_Head.csv
			.
			.
			.
			Task_Head.csv
		test1
		test2
		.
		.
		.
		TestNumber.txt (stores total number of examples/identified actions)
	THIS FILE

Otherwise, organize code as you see fit

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

batchIndex = 0

arch = FLAGS.arch
numberClasses = 11
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
	filename = dirname + '\\Data\\TestNumber.txt'
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
		for line in open(dirname + "\\Data\\test" + str(i) + "\\" + FLAGS.source + "_" + file_names_super[0]):
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
		if labels[i].lower() == "y":
			one_hot_labels.append([1,0,0,0,0,0,0,0,0,0,0])
		elif labels[i].lower() == "cat":
			one_hot_labels.append([0,1,0,0,0,0,0,0,0,0,0])
		elif labels[i].lower() == "supine":
			one_hot_labels.append([0,0,1,0,0,0,0,0,0,0,0])
		elif labels[i].lower() == "seated":
			one_hot_labels.append([0,0,0,1,0,0,0,0,0,0,0])
		elif labels[i].lower() == "sumo":
			one_hot_labels.append([0,0,0,0,1,0,0,0,0,0,0])
		elif labels[i].lower() == "mermaid":
			one_hot_labels.append([0,0,0,0,0,1,0,0,0,0,0])
		elif labels[i].lower() == "towel":
			one_hot_labels.append([0,0,0,0,0,0,1,0,0,0,0])
		elif labels[i].lower() == "trunk":
			one_hot_labels.append([0,0,0,0,0,0,0,1,0,0,0])
		elif labels[i].lower() == "wall":
			one_hot_labels.append([0,0,0,0,0,0,0,0,1,0,0])
		elif labels[i].lower() == "pretzel":
			one_hot_labels.append([0,0,0,0,0,0,0,0,0,1,0])
		else: #OOV
			one_hot_labels.append([0,0,0,0,0,0,0,0,0,0,1])
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
			one_hot_labels.append("y")
		elif predictions[i] == 1:
			one_hot_labels.append("cat")
		elif predictions[i] == 2:
			one_hot_labels.append("supine")
		elif predictions[i] == 3:
			one_hot_labels.append("seated")
		elif predictions[i] == 4:
			one_hot_labels.append("sumo")
		elif predictions[i] == 5:
			one_hot_labels.append("mermaid")
		elif predictions[i] == 6:
			one_hot_labels.append("towel")
		elif predictions[i] == 7:
			one_hot_labels.append("trunk")
		elif predictions[i] == 8:
			one_hot_labels.append("wall")
		elif predictions[i] == 9:
			one_hot_labels.append("pretzel")
		else: #OOV
			one_hot_labels.append("oov")
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
		for line in open(dirname + "\\Data\\test" + str(i)+ "\\Task_" + file_names_super[j]):
			row = line.split(',')
			for l in range(0,3):
				activitySum = activitySum + int(row[l])

		jointActivity.append((activitySum,j))

	jointActivity.sort()

	jointIndexActivity = [x[1] for x in jointActivity]

	if refinement_rate == 0:
		return
	
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
			[Arm1xyz, Head1xyz, Foot1xyz, ...] EVENT 10
			[Arm2xyz, Head2xyz, Foot2xyz, ...] EVENT 2
		
		Parameters: None
		Returns:
			nparray labels
			nparray Data
	'''
	data =  np.empty((sum(timeScores), int(bodySize*3*numSections)))
	dataTask = np.empty((sum(timeScores), int(bodySize*3)))
	sample = True
	numTimeStamps = 0
	labels = []
	c=0

	for i in range(0, int(numberTests)):
		#Determine the number of time stamps in this event
		if FLAGS.refinement == "Tailored":
			global file_names 
			file_names = tailor(i, FLAGS.refinement_rate)
		w=0
		for l in range(numTimeStamps,numTimeStamps+timeScores[i]):
			k=0
			h=0
			for j in range(0, bodySize):
				if FLAGS.position:
					fp = open(dirname + "\\Data\\test" + str(i)+ "\\Position_" + file_names[j])
					for n, line in enumerate(fp):
						if n == w:
							row = line.split(',')
							for m in range(0,3):
								data[l][k]= row[m]
								k = k + 1
			
				if FLAGS.velocity:
					fp = open(dirname + "\\Data\\test" + str(i)+ "\\Velocity_" + file_names[j])
					for n, line in enumerate(fp):
						if n == w:
							row = line.split(',')
							for m in range(0,3):
								data[l][k]= row[m]
								k = k + 1
				if FLAGS.task:
					fp = open(dirname + "\\Data\\test" + str(i)+ "\\Task_" + file_names[j])
					for n, line in enumerate(fp):
						if n == w:
							row = line.split(',')
							for m in range(0,3):
								data[l][k]= row[m]
								k = k + 1
				
				#used for graphing task
				fp = open(dirname + "\\Data\\test" + str(i)+ "\\Task_" + file_names[j])
				for n, line in enumerate(fp):
					if n == w:
						row = line.split(',')
						for m in range(0,3):
							dataTask[l][h]= row[m]
							h = h + 1

			for line in open(dirname + "\\Data\\test" + str(i)+ "\\label.csv"):
				temporaryLabel = line.split()
				labels.append(str(temporaryLabel[0]))
			
			w=w+2					
		numTimeStamps = timeScores[i] + numTimeStamps
	fp.close()

	if FLAGS.verbose:
		print("Number of Sections: ", numSections)
		print("Length of Data: ", data[0][:])
		print("data shape: [", int(numberTests), ", ", int(bodySize*maxEntries*3*numSections), "]")
	
	return data, labels, dataTask

def draw(predictions, correctPredictions, dataTask):	
	'''
		Creates graphs that plot the accuracy of predictions for every frame in an action (Blue). Below, on 
		the same image, a graph representing the number of tasks detected across each body part for 
		that each frame (Red). This occurs for each example in the data.
		
		The function will also output a cumulative histogram for each action with more
		than one example (Green). This cumulative histogram overlays the accuracy of every example of a given
		exercise.
	'''

	start = 0	
	dataRecord = np.zeros((11,maxEntries))
	taskRecord = np.zeros((11,maxEntries))
	offset = 0
	#range from 0-10
	timeRecord = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
	
			
	for i in range (0, int(numberTests)):
		graphData = []
		graphDataX = []
		taskData = []
		for j in range (0, timeScores[i]):
			if predictions[start+j] == correctPredictions[start]:
				graphData.append(1)
						
				if (correctPredictions[start] == "y"):
					dataRecord[0][j] = dataRecord[0][j] + 1
				elif (correctPredictions[start] == "cat"):
					dataRecord[1][j] = dataRecord[1][j] + 1
				elif (correctPredictions[start] == "supine"):
					dataRecord[2][j] = dataRecord[2][j] + 1
				elif (correctPredictions[start] == "seated"):
					dataRecord[3][j] = dataRecord[3][j] + 1
				elif (correctPredictions[start] == "sumo"):
					dataRecord[4][j] = dataRecord[4][j] + 1
				elif (correctPredictions[start] == "mermaid"):
					dataRecord[5][j] = dataRecord[5][j] + 1
				elif (correctPredictions[start] == "towel"):
					dataRecord[6][j] = dataRecord[6][j] + 1
				elif (correctPredictions[start] == "trunk"):
					dataRecord[7][j] = dataRecord[7][j] + 1
				elif (correctPredictions[start] == "wall"):
					dataRecord[8][j] = dataRecord[8][j] + 1
				elif (correctPredictions[start] == "pretzel"):
					dataRecord[9][j] = dataRecord[9][j] + 1
				elif (correctPredictions[start] == "oov"):
					dataRecord[10][j] = dataRecord[10][j] + 1

			else:
				graphData.append(0)
				

			totalTask = sum(dataTask[offset+j][:])
			taskData.append(totalTask)
				
			graphDataX.append(j)

		offset = timeScores[i]
				
		if (correctPredictions[start] == "y"):
			timeRecord[0].append(graphDataX[-1])
		elif (correctPredictions[start] == "cat"):
			timeRecord[1].append(graphDataX[-1])
		elif (correctPredictions[start] == "supine"):
			timeRecord[2].append(graphDataX[-1])
		elif (correctPredictions[start] == "seated"):
			timeRecord[3].append(graphDataX[-1])
		elif (correctPredictions[start] == "sumo"):
			timeRecord[4].append(graphDataX[-1])
		elif (correctPredictions[start] == "mermaid"):
			timeRecord[5].append(graphDataX[-1])
		elif (correctPredictions[start] == "towel"):
			timeRecord[6].append(graphDataX[-1])
		elif (correctPredictions[start] == "trunk"):
			timeRecord[7].append(graphDataX[-1])
		elif (correctPredictions[start] == "wall"):
			timeRecord[8].append(graphDataX[-1])
		elif (correctPredictions[start] == "pretzel"):
			timeRecord[9].append(graphDataX[-1])
		elif (correctPredictions[start] == "oov"):
			timeRecord[10].append(graphDataX[-1])
				
		if FLAGS.verbose:
			print(timeScores[i])
			print("My preditions", predictions[start:start + timeScores[i]])
			print ("Correct predictions", correctPredictions[start])
			print("Length of single test time scores: ", graphDataX[-1])
			print(graphData)
				
		width = .99
		plt.subplot(211)
		plt.bar(graphDataX, graphData, width, facecolor='blue')
		plt.xlabel("Frames")
		plt.annotate(str(graphDataX[-1]), xy = (graphDataX[-1], 1))
		plt.ylabel("Accuracy (Bool)")
		plt.title("Test" + str(i) +": " + str(correctPredictions[start]))
		plt.grid(True)
				
		plt.subplot(212)
		plt.bar(graphDataX, taskData, width, facecolor='red')

		plt.tight_layout()
		plt.savefig(newDir +"\\test" + str(i) + str(correctPredictions[start]) + ".png")
		plt.close()
		start = start + timeScores[i]
		
	totalDataX = range(0, maxEntries)
			
	averageTs = []
	for i in range(0,11):
		current_avg = sum(timeRecord[i])/float(len(timeRecord[i]))
		averageTs.append(int(current_avg))

	width = .99
			
	#y
	if len(timeRecord[0]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[0][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Y")
		annotation = "Average Timespan: " + str(averageTs[0])
		plt.annotate(annotation, xy = (averageTs[0], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[0], width, facecolor='cyan')
		plt.savefig(newDir +"\\yTotalData.png")
		plt.close()

	#cat
	if len(timeRecord[1]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[1][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Cat")
		annotation = "Average Timespan: " + str(averageTs[1])
		plt.annotate(annotation, xy = (averageTs[1], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[1], width, facecolor='cyan')

		plt.savefig(newDir +"\\catTotalData.png")
		plt.close()

	#supine
	if len(timeRecord[2]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[2][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Supine")
		annotation = "Average Timespan: " + str(averageTs[2])
		plt.annotate(annotation, xy = (averageTs[2], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[2], width, facecolor='cyan')

		plt.savefig(newDir +"\\supineTotalData.png")
		plt.close()

	#Seated
	if len(timeRecord[3]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[3][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Seated")
		annotation = "Average Timespan: " + str(averageTs[3])
		plt.annotate(annotation, xy = (averageTs[3], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[3], width, facecolor='cyan')

		plt.savefig(newDir +"\\seatedTotalData.png")
		plt.close()

	#sumo
	if len(timeRecord[4]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[4][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Sumo")
		annotation = "Average Timespan: " + str(averageTs[4])
		plt.annotate(annotation, xy = (averageTs[4], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[4], width, facecolor='cyan')

		plt.savefig(newDir +"\\sumoTotalData.png")
		plt.close()

	#mermaid
	if len(timeRecord[5]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[5][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Mermaid")
		annotation = "Average Timespan: " + str(averageTs[5])
		plt.annotate(annotation, xy = (averageTs[5], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[5], width, facecolor='cyan')

		plt.savefig(newDir +"\\mermaidTotalData.png")
		plt.close()

	#towel
	if len(timeRecord[6]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[6][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Towel")
		annotation = "Average Timespan: " + str(averageTs[6])
		plt.annotate(annotation, xy = (averageTs[6], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[6], width, facecolor='cyan')

		plt.savefig(newDir +"\\towelTotalData.png")
		plt.close()
	#trunk
	if len(timeRecord[7]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[7][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Trunk")
		annotation = "Average Timespan: " + str(averageTs[7])
		plt.annotate(annotation, xy = (averageTs[7], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[7], width, facecolor='cyan')

		plt.savefig(newDir +"\\trunkTotalData.png")
		plt.close()

	#wall
	if len(timeRecord[8]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[8][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Wall")
		annotation = "Average Timespan: " + str(averageTs[8])
		plt.annotate(annotation, xy = (averageTs[8], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[8], width, facecolor='cyan')

		plt.savefig(newDir +"\\wallTotalData.png")
		plt.close()

	#Pretzel
	if len(timeRecord[9]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[9][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("Pretzel")
		annotation = "Average Timespan: " + str(averageTs[9])
		plt.annotate(annotation, xy = (averageTs[9], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[9], width, facecolor='cyan')

		plt.savefig(newDir +"\\pretzelTotalData.png")
		plt.close()

	#oov
	if len(timeRecord[10]) > 1:
		plt.subplot(211)
		plt.bar(totalDataX, dataRecord[10][:], width, facecolor='green')
		plt.xlabel("Frames")
		plt.ylabel("Frequency of Hits")
		plt.title("OOV")
		annotation = "Average Timespan: " + str(averageTs[10])
		plt.annotate(annotation, xy = (averageTs[10], 5))
		plt.grid(True)
		
		plt.subplot(212)
		plt.bar(totalDataX, timeRecord[10], width, facecolor='cyan')

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

	data, labels, dataTask = extractData()
	labelText = labels
	labels = oneHot(labels)

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

		if (FLAGS.test):
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print("Final Accuracy on new Data is: ", "{0:.2%}".format(sess.run(accuracy)), '\n')
			predictions = tf.argmax(pred,1).eval()
			predictions = findExercise(predictions) 
			print("My Predictions", predictions, '\n')
			print("Actual Labels", labelText)
		else: 		
			predictions = tf.argmax(pred,1).eval()
			predictions = findExercise(predictions) 
			correctPredictions = tf.argmax(labels,1).eval()
			correctPredictions = findExercise(correctPredictions)
			draw(predictions, correctPredictions, dataTask)



	
#needed in order to call main
if __name__ == '__main__':
	main()