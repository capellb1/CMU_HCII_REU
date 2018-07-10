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

TRAIN_PERCENT = 0.7
TEST_PERCENT = 0.3

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
			nparray shuffledlabels
			nparray shuffledData
	'''
	data =  np.empty((sum(timeScores), int(bodySize*3*numSections)))
	sample = True

	numTimeStamps = 0
	labels = []
	c=0
	for i in range(0, int(numberTests)):
		#Determine the number of time stamps in this event
		w=0

		if FLAGS.refinement == "Tailored":
			global file_names 
			file_names = tailor(i, FLAGS.refinement_rate)

		for l in range(numTimeStamps,numTimeStamps+timeScores[i]):
			k=0
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

			for line in open(dirname + "\\Data\\test" + str(i)+ "\\label.csv"):
				temporaryLabel = line.split()
				labels.append(str(temporaryLabel[0]))
			
			w=w+2					
		numTimeStamps = timeScores[i] + numTimeStamps

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

def partitionData(features, labels):
	'''
		Divides the total data up into training, validation, and test sets.
		Division based off of percentages stored at the top of the code. Accepts arrays
		and returns separated data along with indicator of how many files are in each division

		Accepts:
			nparray features, lables

		Returns:
			nparray trainLabels, trainFeatures, testLabels, testFeatures
			int train, test
	'''
	train = math.floor(float(sum(timeScores)) * TRAIN_PERCENT)
	test = math.ceil(float(sum(timeScores)) * TEST_PERCENT)

	trainLabels = labels[:train]
	trainFeatures = features[:train]

	testLabels = labels[train:train+test]
	testFeatures = features[train:train+test]
	
	if FLAGS.verbose:
		#Output details on the data we are using
		print("Number of Training Cases: ", train)
		resultsFile.write("Number of Training Cases: " + str(train) + '\n')
		print("Training Labels (Randomized): ", trainLabels)
	
		print("Number of Test Cases: ", test)
		resultsFile.write("Number of Test Cases: " + str(test) + '\n')
		print("Test Lables (Randomized): ", testLabels)

	return trainLabels, trainFeatures, train, testLabels, testFeatures, test

if FLAGS.refinement == "Uniform":
	file_names = uniformRefinement()

elif FLAGS.refinement == "None":
	file_names = file_names_super

dirname = os.path.realpath('.')

numSections = calcSections()

bodySize = calcBodySize()

numberTests = calcNumTests()

maxEntries, timeScores = calcMaxEntries()

folderName = writeFolderLabel()


#Open file used to store accuracy scores and any other printed data
newDir = dirname + '\\Models&Results\\' + folderName
if not (os.path.exists(newDir)):
	os.makedirs(newDir)
resultsFile = open(newDir + '\\Results.txt',"w+")
results2File = open(dirname + '\\Models&Results\\totalResults.txt',"a")

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

	print(bodySize)

	data, labels = extractData()
	labels = oneHot(labels)
	trainLabels, trainData, trainNumber, testLabels, testData, testNumber = partitionData(data, labels)

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

	#construct model
	logits = multilayer_perception(X, weights, biases)

	#define loss and optimizer
	regularization = FLAGS.regularization
	regularizationRate = FLAGS.regularization_rate
	lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)) 

	if (regularization == "l1"):
		print('Regularization: L1 \n')
		l1_regularizer = tf.contrib.layers.l1_regularizer(scale=regularizationRate, scope=None)
		trainedWeights = tf.trainable_variables() # all vars of your graph
		regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, trainedWeights)
		lossOp = lossOp + regularization_penalty
	elif (regularization == "L2"):
		print('Regularization: l2 \n')
		l2_regularizer = tf.contrib.layers.l2_regularizer(scale=regularizationRate, scope=None)
		trainedWeights = tf.trainable_variables() # all vars of your graph
		regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, trainedWeights)
		lossOp = lossOp + regularization_penalty
	else:
		print('Regularization: none \n')
		lossOp = lossOp

	optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
	trainOp = optimizer.minimize(lossOp)

	#initialize global variables
	init = tf.global_variables_initializer()

	#initialize arrays for losses
	trainingLoss = []
	
	# 'Saver' op to save and restore all the variables
	if FLAGS.save:
		saver = tf.train.Saver()

	#creating and running session
	with tf.Session() as sess:
		sess.run(init)

		#training cycle
		for epoch in range(epochsTrained):
			global batchIndex 
			batchIndex = 0
			avgCost = 0
			totalBatch = int(trainNumber/batchSize)

			for i in range(totalBatch):
				batchStart, batchEnd = nextBatch(batchSize, trainNumber)
				batchData = trainData[batchStart:batchEnd]
				batchLabels = trainLabels[batchStart:batchEnd]
				
				shuffledData = np.empty(batchData.shape, dtype=batchData.dtype)
				shuffledLabels = batchLabels
				
				permutation = np.random.RandomState(seed=42).permutation(len(batchLabels))
				
				for old_index, new_index in enumerate(permutation):
					shuffledData[new_index] = batchData[old_index]
					shuffledLabels[new_index] = batchLabels[old_index]

				shuffledLabels = np.asarray(shuffledLabels)

				_, c = sess.run([trainOp, lossOp], feed_dict={X: shuffledData, Y: shuffledLabels})
				avgCost += c/totalBatch

			if (epoch % 10 == 0):
				print("Epoch:", '%04d' % (epoch), "cost={:.9f}".format(avgCost))
				
				if FLAGS.verbose:
					resultsFile.write("Epoch: %04d" % (epoch))
					resultsFile.write(" \n Cost={:.9f}".format(avgCost))
				
				trainingLoss.append(avgCost)

		modelPath =  newDir + "\\ExercisePredicter"		
		
		if FLAGS.save:
			saver.save(sess, modelPath)

		print ("Optimization Finished")
		resultsFile.write("Optimization Finished \n")	

		if FLAGS.verbose:
			#display loss over time curve to aid optimization
			plt.ylabel("LogLoss")
			plt.xlabel("Periods")
			plt.title("Logloss vs Periods")
			plt.plot(trainingLoss, label="training")
			plt.legend()
			plt.savefig(newDir +'\\logLoss.png')

	    #test model 
		pred = tf.nn.softmax(logits)
		correctPrediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))

	    #calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))
		print("Final Training Accuracy:", "{0:.2%}".format(accuracy.eval({X: trainData, Y: trainLabels})))
		print("Final Testing Accuracy:", "{0:.2%}".format(accuracy.eval({X: testData, Y: testLabels})))		
		resultsFile.write("Training Accuracy:" + str(accuracy.eval({X: trainData, Y: trainLabels})) + '\n')	
		results2File.write("Training Accuracy:" + str(accuracy.eval({X: trainData, Y: trainLabels})) + '\n')
		results2File.write("Testing Accuracy:" + str(accuracy.eval({X: testData, Y: testLabels})) + '\n')

#needed in order to call main
if __name__ == '__main__':
	main()