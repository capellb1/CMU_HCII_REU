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
tf.app.flags.DEFINE_string('label','Test','This is the name of the test')

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
folderName = FLAGS.label
newDir = dirname + '\\ModelsAndResults\\' + folderName
if not (os.path.exists(newDir)):
	os.makedirs(newDir)

resultsFile = open(newDir + '\\Results.txt',"w+")
modelsFile = open(newDir + '\\Models.py',"w+")

#Read the number of files(events) that the data contains from the TestNumber.txt file
numberTestFiles = open(filename,"r")
#numberTests = numberTestFiles.read()
numberTests = 6
print("Number of Filed Detected: ", numberTests)
resultsFile.write("Number of Filed Detected: " + str(numberTests) + '\n')

#Determine the maximum/longest running event in the group of seperate tests
#used to define size of the arrays
maxEntries = 0
for i in range(0, int(numberTests)):
	numEntries = 0
	for j in range(0,27):
		for line in open(dirname + "\\Data\\test" + str(i) + "\\Position_" + file_names[j]):
			numEntries = numEntries + 1
		if numEntries > maxEntries:
			maxEntries = numEntries	
print("Maximum Number of Entries in a Single Exercise: ", maxEntries)
resultsFile.write("Maximum Number of Entries in a Single Exercise: " + str(maxEntries) + '\n')

#read data from files
#features [event] [body part] [time stamp] [axis]
#i.e [towel][head][0][x] retrieves the X position of the head during the towel event
def extract_data():
	data =  np.empty((int(numberTests),27, maxEntries,3))
	for i in range(0, int(numberTests)):
		for j in range(0, 27):
			k = 0
			for line in open(dirname + "\\Data\\test" + str(i)+ "\\Position_" + file_names[j]):
				row = line.split(',')
				for l in range(0,3):
					data[i][j][k][l] = row[l]
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

def partition_data(features, labels):
	#Divides the total data up into training, validation, and test sets
	#division based off of percentages stored at the top of the code
	train = math.floor(float(numberTests) * TRAIN_PERCENT)
	validation = math.floor(float(numberTests) * VALIDATION_PERCENT)
	test = math.ceil(float(numberTests) * TEST_PERCENT)
	print("Number of Training Cases: ", train)
	resultsFile.write("Number of Training Cases: "+ str(train) + '\n')
	print("Number of Validation Cases: ", validation)
	resultsFile.write("Number of Validation Cases: " + str(validation) +'\n')
	print("Number of Test Cases: ", test)
	resultsFile.write("Number of Test Cases: " + str(test) + '\n')


	trainLabels = labels[:train]
	trainFeatures = features[:train]
	validationLabels = labels[train:train+validation]
	validationFeatures = features[train:train+validation]
	testLabels = labels[validation:validation+test]
	testFeatures = features[validation:validation+test]
	
	#Output details on the data we are using
	print("Number of Training Cases: ", train)
	print("Training Labels (Randomized): ", trainLabels)
	
	print("Number of Validation Cases: ", validation)
	print("Validation Labels (Randomized): ", validationLabels)
	
	print("Number of Test Cases: ", test)
	print("Test Lables (Randomized): ", testLabels)
	
	return trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures

def one_hot(labels):
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

def constructFeatures():
	#creates the features columns used to train the model
	#each feature column corresponds to a single bodypart and is independent of event
	print("Feature Columns Constructed")
	return set ([tf.feature_column.numeric_column(bodyPartFeatures, shape=[maxEntries,3])
		for bodyPartFeatures in bodyParts])
	
def createTrainingFunction (bodyPartFeatures, labels, batch_size, numEpochs = None):
	#wrap the function definition to allow the creation of multiple input functions later in the code
	def my_input(numEpochs = None):
		#first modify the format of the data into a dictionary: organize the position (x,y,z) by their respective bodyparts
		#no longer tied together by event
		featureDictionary = dict()
		for i in range(0,27):
			tempArray = []
			for j in range(0,len(bodyPartFeatures)):
				tempArray.append(bodyPartFeatures[j][i])
			tempArray = np.asarray(tempArray)
			featureDictionary[bodyParts[i]] = tempArray
		
		#create a tf.Dataset to allow for easy entry into the net
		#think of features organized in columns of bodyparts and the labels are rows
		ds = Dataset.from_tensor_slices((featureDictionary, labels))
		
		#set up the batch mechanic (number of datapoints taken per epoch) along with further shuffling the data
		#create the one shot iterator.. actually does the work of selecting which data to train
		ds = ds.batch(batch_size).repeat(numEpochs)
		ds = ds.shuffle(int(numberTests))
		feature_batch, label_batch, = ds.make_one_shot_iterator().get_next()
		print(feature_batch)
		return feature_batch, label_batch
	return my_input

def createPredictFunction (bodyPartFeatures, labels, batch_size):	
	#wrap the function definition to allow the creation of multiple input functions later in the code
	def create_predict_fn():
		#same as above, change data into a dictionary
		featureDictionary = dict()
		for i in range(0,27):
			tempArray = []
			for j in range(0,len(bodyPartFeatures)):
				tempArray.append(bodyPartFeatures[j][i])
			tempArray = np.asarray(tempArray)
			featureDictionary[bodyParts[i]] = tempArray
		
		#same as above except there is no epoch mechanic because it is being used for predictions (dont want to repeat data being predicted)
		ds = Dataset.from_tensor_slices((featureDictionary, labels))
		ds = ds.batch(batch_size) 
		ds = ds.shuffle(int(numberTests))
		feature_batch, label_batch, = ds.make_one_shot_iterator().get_next()

		return feature_batch, label_batch
	return create_predict_fn

def train(hiddenUnits, steps, trainFeatures, trainLabels, vFeatures, vLabels):
	#define flags as variables
	numEpochs = FLAGS.epochs 
	batchSize = FLAGS.batch_size
	learningRate = FLAGS.learning_rate
	#regularizationRate = FLAGS.regularization_rate
	
	periods = 10
	stepsPerPeriod = steps/periods
	
	#Create the input and predict functions
	predictTrainFunction = createPredictFunction(trainFeatures, trainLabels, batchSize)
	predictValidationFunction = createPredictFunction(vFeatures, vLabels, batchSize)
	trainingFunction = createTrainingFunction(trainFeatures, trainLabels, batchSize, numEpochs)
	print("Prediction and Training Input Functions Created")
	featureColumns = constructFeatures()
	
	#Define the optimizer
	my_optimizer = tf.train.AdagradOptimizer(learning_rate = learningRate)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
	
	#Define the model
	classifier = tf.estimator.DNNClassifier(feature_columns = featureColumns, n_classes = 11, hidden_units = hiddenUnits, optimizer = my_optimizer, config = tf.estimator.RunConfig(keep_checkpoint_max = 1))

	print ("Training model...")
	print ("LogLoss error (on validation data):")
	resultsFile.write("Training model...\n")
	resultsFile.write("LogLoss error (on validation data):\n")
	
	#create lists to store the error over time
	training_errors = []
	validation_errors = []
	
	#convert from encoding to one hot array 
	trainLabels = oneHotArray(trainLabels)
	vLabels2 = oneHotArray(vLabels)
	
	#Perform the training in steps so we can visualize progression of error
	for period in range (0, periods):
		classifier.train(
			input_fn = trainingFunction, steps = stepsPerPeriod)
		
		#Store Metrics about training data and predictions
		training_predictions = list(classifier.predict(input_fn = predictTrainFunction))
		training_probabilities = np.array([item['probabilities'] for item in training_predictions])
		training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
		training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 11)

		#Store metrics about validation data and predictions
		validation_predictions = list(classifier.predict(input_fn = predictValidationFunction))
		validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
		validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
		validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 11)

		#Calculate overall log loss
		training_log_loss = metrics.log_loss(trainLabels, training_pred_one_hot)
		validation_log_loss = metrics.log_loss(vLabels2, validation_pred_one_hot)

		print(" period %02d: %0.2f" % (period, validation_log_loss))
		resultsFile.write(" period %02d: %0.2f" % (period, validation_log_loss) + '\n')

		#store the training error
		
		#store the training error and validation error
		training_errors.append(training_log_loss)
		validation_errors.append(validation_log_loss)
	print("Model training finished")
	resultsFile.write("Model training finished \n")
	
	#tbh not sure what this is doing but seems important
	_ = map(os.remove, glob.glob(os.path.join(classifier.model_dir,'events.out.tfevents*')))

	#store metrics about validation data and predictions
	final_predictions = classifier.predict(input_fn = predictValidationFunction)
	final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

	#Determine overall validation accuracy
	accuracy = metrics.accuracy_score(vLabels, final_predictions)
	print("Final accuracy (on validation data): %0.2f" % accuracy)
	print("Training Errors: ", training_errors)
	resultsFile.write("Final accuracy (on validation data): %0.2f" % accuracy + '\n')
	resultsFile.write("Training Errors: " + str(training_errors) + '\n')

	#display loss over time curve to aid optimization
	plt.ylabel("LogLoss")
	plt.xlabel("Periods")
	plt.title("Logloss vs Periods")
	plt.plot(training_errors, label="training")
	plt.plot(validation_errors, label="validation")
	plt.legend()
	plt.savefig(newDir +'\\logLoss.png')
	plt.show()

	#display confusion matrix to aid in debugging
	cm = metrics.confusion_matrix(vLabels, final_predictions)
	cm_normalized = cm.astype("float") / cm.sum(axis=1) [:, np.newaxis]
	ax = sns.heatmap(cm_normalized, cmap = "bone_r")
	ax.set_aspect(1)
	plt.title("Confusion matrix")
	plt.ylabel("True label")
	plt.xlabel ("Predicted label")
	plt.savefig(newDir + '\\confusionMatrix.png')
	plt.show()
	#confusing because not sure what each label means (0,1,2 etc)

	return classifier


def main(argv = None):
	#Call all methods defined above and determine the shape of the network
	features, labels = extract_data()
	labels= one_hot(labels)
	trainLabels, trainFeatures, vLabels, vFeatures, testLabels, testFeatures = partition_data(features, labels)
	classifier = train(hiddenUnits, 100, trainFeatures, trainLabels, vFeatures, vLabels)
	print("--Training Complete--")
	resultsFile.write("--Training Complete-- '\n")
	predict_test_input_fn = createPredictFunction(testFeatures, testLabels, 100)

	#store and log data on test predictions
	test_predictions = classifier.predict(input_fn=predict_test_input_fn)
	test_predictions = np.array([item['class_ids'][0] for item in test_predictions])

	#Determine test accuracy
	accuracy = metrics.accuracy_score(testLabels, test_predictions)
	print("Final accuracy (on test data): %0.2f" % accuracy)
	resultsFile.write("Final accuracy (on test data): %0.2f" % accuracy +'\n')

	#Maybe Exporting Model
	featureColumnsExport = constructFeatures()
	feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
	export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
	classifier.export_savedmodel( newDir, export_input_fn)



#needed in order to call main
if __name__ == '__main__':
	main()
	
	
	
	
	
	
	

	
