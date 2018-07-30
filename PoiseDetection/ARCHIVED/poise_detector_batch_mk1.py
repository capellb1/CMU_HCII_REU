'''
CMU HCII REU Summer 2018
PI: Dr. Sieworek
Students:  Blake Capella & Deepak Subramanian
Date: 06/26/18

The following code trains a neural net by grouping a set number of frames together as a single feature.
This number of frames is variable, and is controlled by the frames flag.
The source of the data being used to train the net can be toggled between the natural data (Position) and synthetic/calculated
features (Position, Task). This is controlled by the --source flag.

Many flags might not be used in this file, they were included for consistency between the multiple training files.
	poise_detector_mk*.py
	poise_detector_batch_mk*.py
	exercise_detection_mk*.py

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

#Define Flags to change the Hyperperameters
tf.app.flags.DEFINE_integer('batch_size',1000,'number of randomly sampled images from the training set')
tf.app.flags.DEFINE_float('learning_rate',0.001,'how quickly the model progresses along the loss curve during optimization')
tf.app.flags.DEFINE_integer('epochs',100,'number of passes over the training data')
tf.app.flags.DEFINE_float('regularization_rate',0.01,'Strength of regularization')
tf.app.flags.DEFINE_string('regularization', 'Default', 'This is the regularization function used in cost calcuations')
tf.app.flags.DEFINE_string('activation', 'Default', 'This is the activation function to use in the layers')
tf.app.flags.DEFINE_string('label', 'test1', 'This is the label name where the files are saved')
tf.app.flags.DEFINE_integer('frames', 5, 'Number of frames to be analyzed at a time')
tf.app.flags.DEFINE_string('source', 'Position', 'What files to draw data from (Task, Velocity, Position)')
tf.app.flags.DEFINE_string('arch', 'method1', 'This specifies the architecture used')


FLAGS = tf.app.flags.FLAGS

TRAIN_PERCENT = 0.7
VALIDATION_PERCENT = 0.2
TEST_PERCENT = 0.1

#store file names
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
'HipRight.csv',
'KneeRight.csv',    
'AnkleRight.csv',   
'FootRight.csv',     
'HipLeft.csv', 
'KneeLeft.csv',
'AnkleLeft.csv',     
'FootLeft.csv']

#store bodypart names without the file extensions
bodyParts =[
'Head',   
'Neck',    
'SpineShoulder', 
'SpineMid',
'SpineBase',    
'ShoulderRight', 
'ShoulderLeft',  
'HipRight',
'HipLeft', 
'ElbowRight',    
'WristRight',    
'HandRight',     
'HandTipRight',  
'ThumbRight',   
'ElbowLeft',     
'WristLeft',     
'HandLeft',     
'HandTipLeft',  
'ThumbLeft',    
'HipRight',
'KneeRight',    
'AnkleRight',   
'FootRight',     
'HipLeft', 
'KneeLeft',
'AnkleLeft',     
'FootLeft']

#Set the read path to the data folder of the current working directory (allows github upload of data)
dirname = os.path.realpath('.')
filename = dirname + '\\Data\\TestNumber.txt'

#Creating a folder to save the results
folderName = FLAGS.label + "E" + str(FLAGS.epochs) + "LR" + str(FLAGS.learning_rate) + FLAGS.activation + FLAGS.regularization + "RR" + str(FLAGS.regularization_rate) + FLAGS.arch
newDir = dirname + '\\Models&Results\\' + folderName
if not (os.path.exists(newDir)):
	os.makedirs(newDir)

resultsFile = open(newDir + '\\Results.txt',"w+")

#Read the number of files(events) that the data contains from the TestNumber.txt file
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()
print("Number of Filed Detected: ", numberTests)
resultsFile.write("Number of Filed Detected: " + str(numberTests) + '\n')

arch = FLAGS.arch

#GLOBAL
#network parameters:
hiddenLayer1 = 30
hiddenLayer2 = 30
hiddenLayer3 = 30
hiddenLayer4 = 30
numberClasses = 11

#batch Index variable
batchIndex = 0

#Determine the maximum/longest running event in the group of seperate tests
#used to define size of the arrays
maxEntries = 0
timeScores = []
for i in range(0,int(numberTests)):
	numEntries = 0
	for line in open(dirname + "\\Data\\test" + str(i) + "\\" + FLAGS.source + "_" + file_names[0]):
		numEntries = numEntries + 1
	if numEntries > maxEntries:
		maxEntries = numEntries	
	timeScores.append(numEntries)
print("Maximum Number of Entries in a Single Exercise: ", maxEntries)
resultsFile.write("Maximum Number of Entries in Single Exercise: " + str(maxEntries) + '\n')

#read data from files into a flattened array
#each time stamp is a single row and has a corresponding event label... the row containse the xyz for each bodypart
#[Arm1xyz, Head1xyz, Foot1xyz, ...] EVENT 10
#[Arm2xyz, Head2xyz, Foot2xyz, ...] EVENT 2

def extractData():
	data =  np.empty((int((sum(timeScores))/FLAGS.frames), int(FLAGS.frames*27*3)))
	numTimeStamps = 0
	
	for i in range(0, int(numberTests)):
		#Determine the number of time stamps in this event
		for l in range(numTimeStamps,numTimeStamps+timeScores[i]):
			if l%FLAGS.frames == 0:
				k=0
			w=0
			for j in range(0, 27):
				fp = open(dirname + "\\Data\\test" + str(i)+ "\\" + FLAGS.source + "_" + file_names[j])
				for n, line in enumerate(fp):
					if n == w:
						row = line.split(',')
						for m in range(0,3):
							data[l][k]= row[m]
							k = k + 1
			w = w+1

	fp.close()
	labels = []
	#seperate the label from the name and event number stored within the label.csv file(s)
	for i in range (0, int(numberTests)):
		for line in open(dirname + "\\Data\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
			
			for j in range(0,int(timeScores[i]/FLAGS.frames)):
				labels.append(str(temporaryLabel[0]))
	
	#shuffle the data
	shuffledData = np.empty(data.shape, dtype=data.dtype)
	shuffledLabels = labels
	permutation = np.random.permutation(len(labels))
	for old_index, new_index in enumerate(permutation):
		shuffledData[new_index] = data[old_index]
		shuffledLabels[new_index] = labels[old_index]

	shuffledLabels = np.asarray(shuffledLabels)
	return shuffledData, shuffledLabels

def partitionData(features, labels):
	#Divides the total data up into training, validation, and test sets
	#division based off of percentages stored at the top of the code
	train = math.floor(float(sum(timeScores)/FLAGS.frames) * TRAIN_PERCENT)
	validation = math.floor(float(sum(timeScores)/FLAGS.frames) * VALIDATION_PERCENT)
	test = math.ceil(float(sum(timeScores)/FLAGS.frames) * TEST_PERCENT)

	trainLabels = labels[:train]
	trainFeatures = features[:train]
	validationLabels = labels[train:train+validation]
	validationFeatures = features[train:train+validation]
	testLabels = labels[validation:validation+test]
	testFeatures = features[validation:validation+test]
	
	#Output details on the data we are using
	print("Number of Training Cases: ", train)
	resultsFile.write("Number of Training Cases: " + str(train) + '\n')
	print("Training Labels (Randomized): ", trainLabels)
	
	print("Number of Validation Cases: ", validation)
	resultsFile.write("Number of Validation Cases: " + str(validation) + '\n')
	print("Validation Labels (Randomized): ", validationLabels)
	
	print("Number of Test Cases: ", test)
	resultsFile.write("Number of Test Cases: " + str(test) + '\n')
	print("Test Lables (Randomized): ", testLabels)
	
	return trainLabels, trainFeatures, train, validationLabels, validationFeatures, validation, testLabels, testFeatures, test

def oneHot(labels):
	#give each exercise a single numeric representation
	#necessary for converting to tf.DataFrame
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


#creates the model
def multilayer_perception(x, weights, biases):
	activation = FLAGS.activation
	if (arch == "method1" and activation == "Sigmoid"):
		print('Activation Layer: sigmoid \n Architecture Used: method1 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method1" and activation == "Tanh"):
		print('Activation Layer: tanh \n Architecture Used: method1 \n ')
		#Layers
		layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.tanh(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method1" and activation == "Relu"):
		print('Activation Layer: relu \n Architecture Used: method1 \n ')
		#Layers
		layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method1" and activation == "Default"):
		print('Activation Layer: none \n Architecture Used: method1 \n ')
		#Layers
		layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
		layer3 = tf.add(tf.matmul(layer2, weights['h3']), biases['b3'])
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method2" and activation == "Sigmoid"):
		print('Activation Layer: sigmoid \n Architecture Used: method2 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		outLayer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method2" and activation == "Tanh"):
		print('Activation Layer: tanh \n Architecture Used: method2 \n')
		#Layers
		layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		outLayer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method2" and activation == "Relu"):
		print('Activation Layer: relu \n Architecture Used: method2 \n')
		#Layers
		layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		outLayer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method2" and activation == "Default"):
		print('Activation Layer: none \n Architecture Used: method2 \n')
		#Layers
		layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
		outLayer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method3" and activation == "Sigmoid"):
		print('Activation Layer: sigmoid \n Architecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method3" and activation == "Tanh"):
		print('Activation Layer: tanh \n Architecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method3" and activation == "Relu"):
		print('Activation Layer: relu \n Architecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer
	elif (arch == "method3" and activation == "Default"):
		print('Activation Layer: none \n Architecture Used: method3 \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer

def nextBatch(batchSize, trainNumber):
	global batchIndex
	start = batchIndex
	batchIndex += batchSize
	if batchIndex > trainNumber:
		batchIndex = trainNumber
	end = batchIndex
	return start, end

def main(argv = None):
	#Call all methods defined above and determine the shape of the network
	learningRate = FLAGS.learning_rate
	epochsTrained = FLAGS.epochs
	batchSize = FLAGS.batch_size
	#display step

	data, labels = extractData()
	labelText = labels
	labels = oneHot(labels)

	inputLayer = FLAGS.frames*27*3

	init = tf.global_variables_initializer()

	#tf Graph input
	X = tf.placeholder(data.dtype, [None, inputLayer])
	Y = tf.placeholder(labels.dtype, [None, numberClasses])

	if (arch == 'method1'):
		weights = {
		'h1' : tf.Variable(tf.zeros([inputLayer, hiddenLayer1], dtype=data.dtype, name='h1')),
		'h2' : tf.Variable(tf.zeros([hiddenLayer1, hiddenLayer2], dtype=data.dtype, name ='h2')),
		'h3' : tf.Variable(tf.zeros([hiddenLayer2, hiddenLayer3], dtype=data.dtype, name ='h3')),
		'out' : tf.Variable(tf.zeros([hiddenLayer3, numberClasses], dtype=data.dtype, name = 'out'))
		}

		biases = {
		'b1' : tf.Variable(tf.zeros([hiddenLayer1], dtype=data.dtype, name = 'b1')),
		'b2' : tf.Variable(tf.zeros([hiddenLayer2], dtype=data.dtype, name = 'b2')),
		'b3' : tf.Variable(tf.zeros([hiddenLayer3], dtype=data.dtype, name = 'b3')),
		'out' : tf.Variable(tf.zeros([numberClasses], dtype=data.dtype, name = 'outb'))
		}
	elif (arch == "method2"):
		weights = {
		'h1' : tf.Variable(tf.zeros([inputLayer, hiddenLayer1], dtype=data.dtype, name='h1')),
		'h2' : tf.Variable(tf.zeros([hiddenLayer1, hiddenLayer2], dtype=data.dtype, name='h2')),
		'out' : tf.Variable(tf.zeros([hiddenLayer2, numberClasses], dtype=data.dtype, name='out'))
		}

		biases = {
		'b1' : tf.Variable(tf.zeros([hiddenLayer1], dtype=data.dtype, name = 'b1')),
		'b2' : tf.Variable(tf.zeros([hiddenLayer2], dtype=data.dtype, name = 'b2')),
		'out' : tf.Variable(tf.zeros([numberClasses], dtype=data.dtype, name = 'outb'))
		}
	else:
		weights = {
		'h1' : tf.Variable(tf.zeros([inputLayer, hiddenLayer1], dtype=data.dtype, name='h1')),
		'h2' : tf.Variable(tf.zeros([hiddenLayer1, hiddenLayer2], dtype=data.dtype, name='h2')),
		'h3' : tf.Variable(tf.zeros([hiddenLayer2, hiddenLayer3], dtype=data.dtype, name='h3')),
		'h4' : tf.Variable(tf.zeros([hiddenLayer3, hiddenLayer4], dtype=data.dtype, name='h4')),
		'out' : tf.Variable(tf.zeros([hiddenLayer4, numberClasses], dtype=data.dtype, name='out'))
		}

		biases = {
		'b1' : tf.Variable(tf.zeros([hiddenLayer1], dtype=data.dtype, name = 'b1')),
		'b2' : tf.Variable(tf.zeros([hiddenLayer2], dtype=data.dtype, name = 'b2')),
		'b3' : tf.Variable(tf.zeros([hiddenLayer3], dtype=data.dtype, name = 'b3')),
		'b4' : tf.Variable(tf.zeros([hiddenLayer4], dtype=data.dtype, name = 'b4')),		
		'out' : tf.Variable(tf.zeros([numberClasses], dtype=data.dtype, name = 'bout'))
		}

	saver = tf.train.Saver()


	modelPath =  newDir + "\\ExercisePredicter"
	with tf.Session() as sess:

		saver.restore(sess, modelPath )

		sess.run(weights)
		sess.run(biases)
		
		logits = multilayer_perception(data, sess.run(weights), sess.run(biases))
		pred = tf.nn.softmax(logits)

		if (FLAGS.mode == "test"):
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print("Final Accuracy on new Data is: ", "{0:.2%}".format(sess.run(accuracy)), '\n')
			predictions = tf.argmax(pred,1).eval()
			predictions = findExercise(predictions) 
			print("My predictions", predictions, '\n')
			print("Original predictions", labelText)
		else: 		
			predictions = tf.argmax(pred,1).eval()
			predictions = findExercise(predictions) 
			print("My preditions", predictions)
		
	
#needed in order to call main
if __name__ == '__main__':
	main()