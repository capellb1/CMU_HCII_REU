#CMU HCII REU Summer 2018
#PI: Dr. Sieworek
#Students:  Blake Capella & Deepak Subramanian
#
#MUST HAVE AT LEAST 5 files

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
tf.app.flags.DEFINE_integer('batch_size',10,'number of randomly sampled images from the training set')
tf.app.flags.DEFINE_float('learning_rate',0.001,'how quickly the model progresses along the loss curve during optimization')
tf.app.flags.DEFINE_integer('epochs',10,'number of passes over the training data')
tf.app.flags.DEFINE_float('regularization_rate',0.01,'Strength of regularization')
tf.app.flags.DEFINE_string('regularization', 'Default', 'This is the regularization function used in cost calcuations')
tf.app.flags.DEFINE_string('activation', 'Default', 'This is the activation function to use in the layers')
tf.app.flags.DEFINE_string('label', 'test1', 'This is the label name where the files are saved')

FLAGS = tf.app.flags.FLAGS

TRAIN_PERCENT = 0.7
VALIDATION_PERCENT = 0.2
TEST_PERCENT = 0.1
hiddenUnits = [16, 16]

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
folderName = FLAGS.label + "E" + str(FLAGS.epochs) + "LR" + str(FLAGS.learning_rate) + FLAGS.activation + FLAGS.regularization + "RR" + str(FLAGS.regularization_rate)
newDir = dirname + '\\ModelsAndResults_mk3\\' + folderName
if not (os.path.exists(newDir)):
	os.makedirs(newDir)

resultsFile = open(newDir + '\\Results.txt',"w+")

#Read the number of files(events) that the data contains from the TestNumber.txt file
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()
print("Number of Filed Detected: ", numberTests)
#resultsFile.write("Number of Filed Detected: " + str(numberTests) + '\n')


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
for i in range(0,int(numberTests)):
	numEntries = 0
	for j in range(0,27):
		for line in open(dirname + "\\Data\\test" + str(i) + "\\Position_" + file_names[j]):
			numEntries = numEntries + 1
		if numEntries > maxEntries:
			maxEntries = numEntries	
print("Maximum Number of Entries in a Single Exercise: ", maxEntries)
#resultsFile.write("Maximum Number of Entries in Single Exercise: " + str(maxEntries) + '\n')

#read data from files
#features [event] [body part] [time stamp] [axis]
#i.e [towel][head][0][x] retrieves the X position of the head during the towel event

def extractData():
	data =  np.empty((int(numberTests), int(26*maxEntries*3)))

	for i in range(0, int(numberTests)):
		k = 0
		for j in range(0, 27):
			for line in open(dirname + "\\Data\\test" + str(i)+ "\\Position_" + file_names[j]):
				row = line.split(',')
				for l in range(0,3):
					data[i][k] = row[l]
					k = k +1
	labels = []
	#seperate the label from the name and event number stored within the label.csv file(s)
	for i in range (0, int(numberTests)):
		for line in open(dirname + "\\Data\\test" + str(i)+ "\\label.csv"):
			temporaryLabel = line.split()
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
	train = math.floor(float(numberTests) * TRAIN_PERCENT)
	validation = math.floor(float(numberTests) * VALIDATION_PERCENT)
	test = math.ceil(float(numberTests) * TEST_PERCENT)

	trainLabels = labels[:train]
	trainFeatures = features[:train]
	validationLabels = labels[train:train+validation]
	validationFeatures = features[train:train+validation]
	testLabels = labels[validation:validation+test]
	testFeatures = features[validation:validation+test]
	
	#Output details on the data we are using
	print("Number of Training Cases: ", train)
	#resultsFile.write("Number of Training Cases: " + str(train) + '\n')
	print("Training Labels (Randomized): ", trainLabels)
	
	print("Number of Validation Cases: ", validation)
	#resultsFile.write("Number of Validation Cases: " + str(validation) + '\n')
	print("Validation Labels (Randomized): ", validationLabels)
	
	print("Number of Test Cases: ", test)
	#resultsFile.write("Number of Test Cases: " + str(test) + '\n')
	print("Test Lables (Randomized): ", testLabels)
	
	return trainLabels, trainFeatures, train, validationLabels, validationFeatures, validation, testLabels, testFeatures, test

def oneHot(labels):
	#give each exercise a single numeric representation
	#necessary for converting to tf.DataFrame
	one_hot_labels = []
	for i in range(0,len(labels)):
		if labels[i].lower() == "y":
			one_hot_labels.append([0])
		elif labels[i].lower() == "cat":
			one_hot_labels.append([1])
		elif labels[i].lower() == "supine":
			one_hot_labels.append([2])
		elif labels[i].lower() == "seated":
			one_hot_labels.append([3])
		elif labels[i].lower() == "sumo":
			one_hot_labels.append([4])
		elif labels[i].lower() == "mermaid":
			one_hot_labels.append([5])
		elif labels[i].lower() == "towel":
			one_hot_labels.append([6])
		elif labels[i].lower() == "trunk":
			one_hot_labels.append([7])
		elif labels[i].lower() == "wall":
			one_hot_labels.append([8])
		elif labels[i].lower() == "pretzel":
			one_hot_labels.append([9])
		else: #OOV
			one_hot_labels.append([10])
	one_hot_labels = np.asarray(one_hot_labels)
	print("Lable Encoding Complete")
	return one_hot_labels

def oneHotArray(labels):
	#convert the integer one hot encoding into a binary array
	#necessary for actual training of the net
	one_hot_labels = []
	for i in range(0,len(labels)):
		if labels[i] == 0:
			one_hot_labels.append([1,0,0,0,0,0,0,0,0,0,0])
		elif labels[i] == 1:
			one_hot_labels.append([0,1,0,0,0,0,0,0,0,0,0])
		elif labels[i] == 2:
			one_hot_labels.append([0,0,1,0,0,0,0,0,0,0,0])
		elif labels[i] == 3:
			one_hot_labels.append([0,0,0,1,0,0,0,0,0,0,0])
		elif labels[i] == 4:
			one_hot_labels.append([0,0,0,0,1,0,0,0,0,0,0])
		elif labels[i] == 5:
			one_hot_labels.append([0,0,0,0,0,1,0,0,0,0,0])
		elif labels[i] == 6:
			one_hot_labels.append([0,0,0,0,0,0,1,0,0,0,0])
		elif labels[i] == 7:
			one_hot_labels.append([0,0,0,0,0,0,0,1,0,0,0])
		elif labels[i] == 8:
			one_hot_labels.append([0,0,0,0,0,0,0,0,1,0,0])
		elif labels[i] == 9:
			one_hot_labels.append([0,0,0,0,0,0,0,0,0,1,0])
		else: #OOV
			one_hot_labels.append([0,0,0,0,0,0,0,0,0,0,1])
	one_hot_labels = np.asarray(one_hot_labels)
	print("One Hot Encoding Complete")
	return one_hot_labels

#creates the model
def multilayer_perception(x, weights, biases):
	activation = FLAGS.activation
	if (activation == "Sigmoid"):
		print('Activation Layer: sigmoid \n')
		#Layers
		layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	elif (activation == "Tanh"):
		print('Activation Layer: tanh \n')
		#Layers
		layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.tanh(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		layer4 = tf.nn.tanh(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))		
		outLayer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
		return outLayer
	elif (activation == "Relu"):
		print('Activation Layer: relu \n')
		#Layers
		layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
		layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
		return outLayer
	else:
		print('Activation Layer: none \n')
		#Layers
		layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
		layer3 = tf.add(tf.matmul(layer2, weights['h3']), biases['b3'])
		outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])
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
	labels = oneHot(labels)
	labels = oneHotArray(labels)
	trainLabels, trainData, trainNumber, validationLabels, validationData, validationNumber, testLabels, testData, testNumber = partitionData(data, labels)

	inputLayer = 26*maxEntries*3

	#tf Graph input
	X = tf.placeholder(data.dtype, [None, inputLayer])
	Y = tf.placeholder(labels.dtype, [None, numberClasses])

	#store layers weight & bias
	weights = {
		'h1' : tf.Variable(tf.random_normal([inputLayer, hiddenLayer1], dtype=data.dtype)),
		'h2' : tf.Variable(tf.random_normal([hiddenLayer1, hiddenLayer2], dtype=data.dtype)),
		'h3' : tf.Variable(tf.random_normal([hiddenLayer2, hiddenLayer3], dtype=data.dtype)),
		'h4' : tf.Variable(tf.random_normal([hiddenLayer3, hiddenLayer4], dtype=data.dtype)),
		'out' : tf.Variable(tf.random_normal([hiddenLayer4, numberClasses], dtype=data.dtype))
	}

	biases = {
		'b1' : tf.Variable(tf.random_normal([hiddenLayer1], dtype=data.dtype)),
		'b2' : tf.Variable(tf.random_normal([hiddenLayer2], dtype=data.dtype)),
		'b3' : tf.Variable(tf.random_normal([hiddenLayer3], dtype=data.dtype)),
		'b4' : tf.Variable(tf.random_normal([hiddenLayer4], dtype=data.dtype)),		
		'out' : tf.Variable(tf.random_normal([numberClasses], dtype=data.dtype))
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
		l1_regularizer = tf.contrib.layers.l2_regularizer(scale=regularizationRate, scope=None)
		trainedWeights = tf.trainable_variables() # all vars of your graph
		regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, trainedWeights)
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

				_, c = sess.run([trainOp, lossOp], feed_dict={X: batchData, Y: batchLabels})
				avgCost += c/totalBatch

			if (epoch % 10 == 0):
				print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avgCost))
				resultsFile.write("Epoch: %04d" % (epoch+1))
				#resultsFile.write(" \n Cost={:.9f}".format(avgCost))
				trainingLoss.append(avgCost)
				#test model 
				pred = tf.nn.softmax(logits)
				correctPrediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
				accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))
				print("Validation Accuracy:", accuracy.eval({X: validationData, Y: validationLabels}))
				resultsFile.write("Validation Accuracy:" + str(accuracy.eval({X: validationData, Y: validationLabels})) + '\n')	

		modelPath =  newDir + "\\model.ckpt"

		print ("Optimization Finished")
		#resultsFile.write("Optimization Finished \n")	

		#display loss over time curve to aid optimization

	    #test model 
		pred2 = tf.nn.softmax(logits)
		correctPrediction2 = tf.equal(tf.argmax(pred2,1), tf.argmax(Y,1))

	    #calculate accuracy
		accuracy2 = tf.reduce_mean(tf.cast(correctPrediction2, "float"))

		print("Training Accuracy:", accuracy.eval({X: trainData, Y: trainLabels}))
		resultsFile.write("Training Accuracy:" + str(accuracy.eval({X: trainData, Y: trainLabels})) + '\n')	
		print("Final Validation Accuracy:", accuracy.eval({X: validationData, Y: validationLabels}))
		resultsFile.write("Final Validation Accuracy:" + str(accuracy.eval({X: validationData, Y: validationLabels})) + '\n')	
		print("Testing Accuracy:", accuracy.eval({X: testData, Y: testLabels}))
		resultsFile.write("Testing Accuracy:" + str(accuracy.eval({X: testData, Y: testLabels})) + '\n')	

#needed in order to call main
if __name__ == '__main__':
	main()
	
	
	
	
	
	
	

	
