#CMU HCII REU Summer 2018
#PI: Dr. Sieworek
#Students:  Blake Capella & Deepak Subramanian
#
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.data import Dataset
import numpy as np
import pandas as pd
import math 



#Define Flags
tf.app.flags.DEFINE_integer('batch_size',10,'number of randomly sampled images from the training set')
tf.app.flags.DEFINE_float('learning_rate',0.01,'how quickly the model progresses along the loss curve during optimization')
tf.app.flags.DEFINE_integer('epochs',10,'number of passes over the training data')
tf.app.flags.DEFINE_float('regularization_rate',0.01,'Strength of regularization')
FLAGS = tf.app.flags.FLAGS

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

#store file names
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

numberTestFiles = open("C:\\Users\Deepak Subramanian\Documents\Internship\HCII Research (2018)\\task_sequencer_v2\Data\\TestNumber.txt", "r")
numberTests = numberTestFiles.read()

#read data from files
#features [event] [body part] [time stamp]

def extract_data():
	labels = []
	data = []
	for k in range(0, int(numberTests)): #iterates through "folders" or entire action sequences
	#for k in range(0, 1): #iterates through "folders" or entire action sequences
		event = []
		for i in range(0,27): #iterates through each body parts position files
			bodypart =[]
			#for line in open("D:\CMU\kinect_data\\test" + str(k)+ "\\Position_" + file_names[i]): #iterates through every time stamp recorded for each bodypart
			for line in open("C:\\Users\Deepak Subramanian\Documents\Internship\HCII Research (2018)\\task_sequencer_v2\Data\\test" + str(k)+ "\\Position_" + file_names[i]):
				row = line.split(',')
				bodypart.append(row[:-1])
			event.append(bodypart)
		#for line in open("D:\CMU\kinect_data\\test" + str(k)+ "\\label.csv"):
		for line in open("C:\\Users\Deepak Subramanian\Documents\Internship\HCII Research (2018)\\task_sequencer_v2\Data\\test" + str(k)+ "\\label.csv"):
			labels.append(line.strip('\n'))
		pair = (labels[k], event)
		data.append(pair)
	
	#features = np.asarray(features)
	#labels = np.asarray(labels)
	return data

def partition_data(extractedData):
	np.random.shuffle(extractedData)

	labels = []
	features = []
	
	for i in range (0,int(numberTests)):
		labels.append(extractedData[i][0])
		features.append(extractedData[i][1])

	#labels = one_hot(labels)

	train = math.floor(float(numberTests) * .7)
	validation = math.floor(float(numberTests) * .2)
	test = math.ceil(float(numberTests) * .1)

	trainLabels = labels[:train]
	trainFeatures = features[:train]
	validationLabels = labels[train:train+validation]
	validationFeatures = features[train:train+validation]
	testLabels = labels[validation:validation+test]
	testFeatures = features[validation:validation+test]

	return trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures

def one_hot(labels):
	one_hot_labels = []
	for i in range(0,len(labels)):
		if(labels[i].lower() == "y"):
			one_hot_labels.append([1,0,0,0,0,0,0,0,0,0,0])
		elif(labels[i].lower() == "cat"):
			one_hot_labels.append([0,1,0,0,0,0,0,0,0,0,0])
		elif(labels[i].lower() == "supine"):
			one_hot_labels.append([0,0,1,0,0,0,0,0,0,0,0])
		elif(labels[i].lower() == "seated"):
			one_hot_labels.append([0,0,0,1,0,0,0,0,0,0,0])
		elif(labels[i].lower() == "sumo"):
			one_hot_labels.append([0,0,0,0,1,0,0,0,0,0,0])
		elif(labels[i].lower() == "mermaid"):
			one_hot_labels.append([0,0,0,0,0,1,0,0,0,0,0])
		elif(labels[i].lower() == "towel"):
			one_hot_labels.append([0,0,0,0,0,0,1,0,0,0,0])
		elif(labels[i].lower() == "trunk"):
			one_hot_labels.append([0,0,0,0,0,0,0,1,0,0,0])
		elif(labels[i].lower() == "wall"):
			one_hot_labels.append([0,0,0,0,0,0,0,0,1,0,0])
		elif(labels[i].lower() == "pretzel"):
			one_hot_labels.append([0,0,0,0,0,0,0,0,0,1,0])
		else: #OOV
			one_hot_labels.append([0,0,0,0,0,0,0,0,0,0,1])
	one_hot_labels = np.asarray(one_hot_labels)
	return one_hot_labels

def constructFeatures():
	return set ([tf.feature_column.numeric_column(bodyPartFeatures)
		for bodyPartFeatures in bodyParts])

def train():
	numEpochs = FLAGS.epochs 
	batchSize = FLAGS.batch_size
	learningRate = FLAGS.learning_rate
	regularizationRate = FLAGS.regularization_rate
	
def my_input(bodyPartData, labels, batch_size, numEpochs):
	#bodyPartDataArray = []
	#for i in range(0,len(bodyPartData)):
		#eventData = []
		#for j in range(0, 27):
			#eventData.append(bodyPartData[j][i])
		#bodyPartDataArray.append(eventData)

	#bodyPartDictionary = dict()
	#for i in range (0,len(bodyPartData)):
		#bodyPartDictionary[i] = bodyPartData[i]
	bodyPartDF = pd.DataFrame(bodyPartData)
	#print(bodyPartDF[0])

	#tempB = []
	#for i in range (0, len(bodyPartData)):
	#	tempB.append(bodyPartData[i])
	#bodyPartDict = dict()
	#bodyPartDict['0'] = tempB
	#bodyPartDF = pd.DataFrame(bodyPartDict)
	#bodyPartDF = bodyPartDF.values
	#print (bodyPartDF)

	#tempA = []
	#for i in range (0, len(labels)):
	#	tempA.append(labels[i])

	#labelsDict = dict()
	#labelsDict['0'] = tempA
	#print (labels)
	#labels2 = []
	#for i in range (0,len(labels)):
		#labels2.append([labels[i]])
	#print (labels2)

	#print(len(bodyPartData[1][0]))

	labelsDF = pd.DataFrame(labels)
	#labelsDF = labelsDF.values

	#print(bodyPartDF)
	#print(labelsDF)
	#features, labels2 = (np.random.sample((10,27)), np.random.sample((3,1)))
	#features2 = [[1,2], [1,2,3], [1,2]]
	#features2 = pd.DataFrame(features2)
	#print (features2)
	#labels2 = pd.DataFrame(labels2)
	#features2 = []
	#for i in range(0, len(features)):
	#	features2.append(features)
	#print(features2, labels2)
	label_tensor = tf.convert_to_tensor(labelsDF)
	#print(bodyPartDF[0][0])
	#print(len(bodyPartDF[0][0]))

	bodyDict = dict()
	bodyDict['0'] = bodyPartDF[0]
	bodyPartDF = pd.DataFrame(bodyDict)
	print (bodyPartDF)
	print(labelsDF)
	feature_tensor = tf.constant(bodyPartDF)
	#print (feature_tensor)


	#feature_tensor = tf.convert_to_tensor(bodyPartDF)
	
	#dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))
	#dataset = tf.data.Dataset.from_tensor_slices((label_tensor, feature_tensor))

	#print(labelsDF)
	#print(bodyPartDF)
	#labels3 = []
	#for i in range(0,len(labels2)):
		#labels3.append(labels2[i])
	#print(labels3)
	#features2 = []
	#for i in range(0,len(features)):
		#labels3.append(features[i])
	#dataset = tf.data.Dataset.from_tensor_slices((features,labels3))

	#print(len(labels))
	#print(len(labels[0]))
	#print (len(bodyPartData))
	#print(len(bodyPartData[0]))

	#print(features)
	#print(labels)
	#print(dataset)
	#print (bodyPartDictionary['Head'][0])
	#print(len(bodyPartData[0]))
	
	#print(len(bodyPartDF))
	#print(bodyPartDF[0])
	#print(labelsDF[0])
	#print(len(labelsDF))
	#ds = Dataset.from_tensor_slices((bodyPartDF[0], labels[0]))

	#ds = ds.batch(batch_size).repeat(numEpochs)
	feature_batch = 1
	label_batch = 1
	#feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
	#return feature_batch, label_batch


def main(argv = None):

	extractedData = extract_data()
	trainLabels, trainFeatures, vLabels, vFeatures, testLabels, testFeatures = partition_data(extractedData)
	featureColumns = constructFeatures();
	my_input(trainFeatures, trainLabels, 1, 1)
	
if __name__ == '__main__':
	main()
	
	
	
	
	
	
	

	
