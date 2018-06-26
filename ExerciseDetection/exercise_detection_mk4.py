import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.contrib import slim
import os
import numpy as np
import math

#Data Visualization Methods
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
import seaborn as sns
import glob

#Show Debugging Output
tf.logging.set_verbosity(tf.logging.DEBUG)

#set default flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
	name='model_dir', default='./Model_Default_Location',
	help='Output directory for model and training stats')

tf.app.flags.DEFINE_string(
	name='data_dir', default='./stats',
	help='Output directory for model and training stats')

tf.app.flags.DEFINE_float(
	name='learn', default = 0.02,
	help='Learning Rate')

tf.app.flags.DEFINE_integer(
	name='step', default=2500,
	help='Total step size / number of iterations to run')
	
tf.app.flags.DEFINE_string(
	name='label', default='Blake_Net',
	help='This is the name of the test')
	
tf.app.flags.DEFINE_integer(
	name='batch', default=10,
	help='Batch Size')
	
tf.app.flags.DEFINE_string(
	name='activation', default='sigmoid',
	help='What activation layer to use (i.e. relu)')
	
tf.app.flags.DEFINE_float(
	name='regularization', default=0.01,
	help='Regularization Rate')



# OLD DATA INPUT CODE #####################################
TRAIN_PERCENT = 0.7
VALIDATION_PERCENT = 0.2
TEST_PERCENT = 0.1
BATCH_SIZE = FLAGS.batch
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
folderName = FLAGS.label + "LR" + str(FLAGS.learn) + "Reg" + str(FLAGS.regularization)
newDir = dirname + '\\Models&Results\\' + folderName
if not (os.path.exists(newDir)):
	os.makedirs(newDir)

resultsFile = open(newDir + '\\Results.txt',"w+")
resultsFile.write("Learning Rate: " + str(FLAGS.learn)+ '\n')
resultsFile.write("Batch Size: " + str(FLAGS.batch)+ '\n')
resultsFile.write("Activation Layer: " + str(FLAGS.activation)+ '\n')
resultsFile.write("Regularization Rate: " + str(FLAGS.regularization)+ '\n')
resultsFile.write("Regularization Type: L1"+ '\n')
resultsFile.write("Net Architecture: 10f,10f,11o" + '\n')

#Read the number of files(events) that the data contains from the TestNumber.txt file
numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()

resultsFile.write("Number of Filed Detected: " + str(numberTests) + '\n')

#Determine the maximum/longest running event in the group of seperate tests
#used to define size of the arrays
maxEntries = 0
for i in range(0, int(numberTests)):
	numEntries = 0
	for line in open(dirname + "\\Data\\test" + str(i) + "\\Position_" + file_names[j]):
		numEntries = numEntries + 1
	if numEntries > maxEntries:
		maxEntries = numEntries	
resultsFile.write("Maximum Number of Entries in a Single Exercise: " + str(maxEntries) + '\n')
#read data from files
#features [event] [body part] [time stamp] [axis]
#i.e [towel][head][0][x] retrieves the X position of the head during the towel event
def extract_data():
	data =  np.empty((int(numberTests),27*maxEntries*3))
	for i in range(0, int(numberTests)):
		k=0
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
	trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures = partition_data(shuffledData, shuffledLabels)
	processed_data = [trainLabels, trainFeatures, validationLabels, validationFeatures, testLabels, testFeatures]
	
	return processed_data

def partition_data(features, labels):
	#Divides the total data up into training, validation, and test sets
	#division based off of percentages stored at the top of the code
	train = math.floor(float(numberTests) * TRAIN_PERCENT)
	validation = math.floor(float(numberTests) * VALIDATION_PERCENT)
	test = math.ceil(float(numberTests) * TEST_PERCENT)
	
	resultsFile.write("Number of Training Cases: "+ str(train) + '\n')
	resultsFile.write("Number of Validation Cases: " + str(validation) +'\n')
	resultsFile.write("Number of Test Cases: " + str(test) + '\n')


	trainLabels = labels[:train]
	trainFeatures = features[:train]
	validationLabels = labels[train:train+validation]
	validationFeatures = features[train:train+validation]
	testLabels = labels[validation:validation+test]
	testFeatures = features[validation:validation+test]
	
	return one_hot(trainLabels), trainFeatures, one_hot(validationLabels), validationFeatures, one_hot(testLabels), testFeatures

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
	return one_hot_labels
# NEW CODE ################################################
#define and run experiment
def run_experiment(argv=None):
	'''run the training experiment'''
	params = tf.contrib.training.HParams(
		learning_rate=FLAGS.learn,
		n_classes=11,
		train_steps=FLAGS.step, #5,000
		min_eval_frequency=100
	)
	
	#set the run_config and the directory to save the moddel and stats
	run_config= tf.estimator.RunConfig()
	run_config= run_config.replace(model_dir=newDir)
	run_config = run_config.replace(
		save_checkpoints_steps=params.min_eval_frequency)
	#define the classifier
	estimator = get_estimator(run_config, params)
	#setup data loaders
	#ORIGINAL -- mnist =tf.contrib.learn.datasets.load_dataset("mnist")
	exercise_data = extract_data()
	train_input_fn, train_input_hook = get_train_inputs(
		batch_size=BATCH_SIZE, excercise_data=exercise_data )
	
	eval_input_fn, eval_input_hook = get_eval_inputs(
		batch_size=BATCH_SIZE, exercise_data=exercise_data )	
	
	test_input_fn, test_input_hook = get_test_inputs(
		batch_size=BATCH_SIZE, exercise_data=exercise_data )	
		
	train_spec = tf.estimator.TrainSpec(
		input_fn=train_input_fn, #First Class Function
		max_steps=params.train_steps,	#minibatch steps
		hooks=[train_input_hook],		#hooks for training
		)
	
	eval_spec = tf.estimator.EvalSpec(
		input_fn=eval_input_fn, 	#First Class Function
		steps=None, #use evaluation feeder until it is empty
		hooks=[eval_input_hook] #hooks for evaluation
		)
	
	test_spec = tf.estimator.EvalSpec(
		input_fn=test_input_fn,
		steps = None,
		hooks = [test_input_hook])
	
	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

	
	#TODO Add Exporter and Serving Function
	#exporter = tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=None)
	
def get_estimator(run_config, params):
	'''Return the model as a tensorflow estimator object
		
		Args:
			run_config (RunConfig): Configuration for Estimator RunConfig
			params (Hparams): Hyperparameters
	'''
	
	return tf.estimator.Estimator(
		model_fn=model_fn, #first class fn
		params=params,
		config=run_config
	)
	
def model_fn(features, labels, mode, params):
	''' Model function used in the estimator
	
		Args:
			features (Tensor): Input features to the model
			labels (Tensor): Label tensor for training and evaluation
			mode (ModeKeys): Specifies if training, evaluation, or prediction
			params (HParams): hyperparameters
		
		returns:
			(EstimatorSpec): model to be run by estimator
	'''
	
	is_training = mode == tf.estimator.ModeKeys.TRAIN
	#Define models achitecture
	logits = architecture(features, is_training=is_training)
	predictions = tf.argmax(logits, axis=-1)
	#Loss, training, and eval operations are not needed during inference
	loss = None
	train_op = None
	eval_metric_ops = {}

	if mode != tf.estimator.ModeKeys.PREDICT:
		loss = tf.losses.sparse_softmax_cross_entropy(
				labels=tf.cast(labels, tf.int32),
				logits=logits)
		train_op = get_train_op_fn(loss,params)
		eval_metric_ops = get_eval_metric_ops(labels, predictions)

	
	return tf.estimator.EstimatorSpec(
		mode=mode,
		predictions=predictions,
		loss=loss,
		train_op = train_op,
		eval_metric_ops = eval_metric_ops
	)

def get_train_op_fn(loss, params):
	'''Get the training Op.
		
		Args:
			loss (Tensor): Scalar Tensor that represents the loss function
			params (HParams): Hyperparameters (needs to have 'learning rate')
			
		Returns:
			Training Op
	'''
	
	return tf.contrib.layers.optimize_loss(
		loss=loss,
		global_step=tf.train.get_global_step(),
		optimizer=tf.train.AdamOptimizer,
		learning_rate=params.learning_rate
		)

def get_eval_metric_ops(labels, predictions):
	'''Return a dict of the evaluation ops
	
	Args:
		labels (Tensor): labels tensor for training and evaluation
		predictions (Tensor): Predictions Tesnor
	
	Returns:
		Dict of metric results keyed by name
	'''
	
	return {
			'Accuracy': tf.metrics.accuracy(
				labels=labels,
				predictions=predictions,
				name='accuracy')
			}
def architecture(inputs, is_training, scope='ExcersizeConvNet'):
	'''Return the output operation following the network architecture
	
		Args:
			inputs (Tensor): Input Tensor
			is_training (Bool): true iff in training mode
			scope (str): Name of the scope of the architecture
		
		Returns:
			Logits of output Op for the network
	'''
	regularizer = tf.contrib.layers.l1_regularizer(scale = FLAGS.regularization)
	if(FLAGS.activation == 'relu'):		
		net = tf.contrib.layers.fully_connected(inputs, 10, weights_regularizer = regularizer)
		net = tf.contrib.layers.fully_connected(net, 10,  weights_regularizer = regularizer)
		net = tf.contrib.layers.fully_connected(net, 11,activation_fn=None)
	
	elif(FLAGS.activation == 'sigmoid'):	
		net = tf.contrib.layers.fully_connected(inputs, 10, activation_fn=tf.sigmoid ,  weights_regularizer = regularizer)
		net = tf.contrib.layers.fully_connected(net, 10, activation_fn=tf.sigmoid,  weights_regularizer = regularizer)
		net = tf.contrib.layers.fully_connected(net, 11,activation_fn=None)
	
	elif(FLAGS.activation == 'tanh'):		
		net = tf.contrib.layers.fully_connected(inputs, 10, activation_fn=tf.tanh,  weights_regularizer = regularizer)
		net = tf.contrib.layers.fully_connected(net, 10, activation_fn=tf.tanh,  weights_regularizer = regularizer)
		net = tf.contrib.layers.fully_connected(net, 11,activation_fn=None)
		
	return net	
		
class IteratorInitializerHook(tf.train.SessionRunHook):
	'''Hook to inialize dta iterator after session is created'''
	
	def __init__(self):
		super(IteratorInitializerHook, self).__init__()
		self.iterator_initializer_func = None
	
	def after_create_session(self, session, coord):
		'''Initialize the iterator after the session has been created'''
		self.iterator_initializer_func(session)

def get_train_inputs(batch_size,excercise_data):
	'''Return the input function to get the training data
	
		Args:
			batch_size (int): Batch size of training iterator that is returned by the input function
			exercise_data (Object): Object holding the loaded excercise data in the form (feaures, labels)
		
		Returns:
			(Input Function, IteratorInitializerHook)
			-Function that returns (features, labels) when called
			-Hook to initialize input iterator
	'''
	iterator_initializer_hook = IteratorInitializerHook()
	def train_inputs():
		'''Returns training set as Operations
		
			Returns:
				(features, labels) Operations that iterate over the dataset on every evaluation
		'''	
	
		with tf.name_scope('Training_data'):
			#Get MINST data
			#ORIGINAL -- images = excercise_data.train.images.reshape([-1,28,28,1])
			images = excercise_data[1]
			labels = excercise_data[0]
			
			#ORIGINAL -- labels = excercise_data.train.labels
			#Define Placeholders
			images_placeholder = tf.placeholder(images.dtype, images.shape)
			labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
			#Build Dataset Iterator
			dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
			dataset = dataset.repeat(None) #infinite iterations (normally would be Epochs)
			dataset = dataset.shuffle(buffer_size =1000)
			dataset = dataset.batch(batch_size)
			iterator = dataset.make_initializable_iterator()
			next_example, next_label = iterator.get_next()
			#Set runhook to initialize iterator
			iterator_initializer_hook.iterator_initializer_func = \
				lambda sess: sess.run(
					iterator.initializer,
					feed_dict={images_placeholder: images,
								labels_placeholder: labels})
			return next_example, next_label
		
	return train_inputs, iterator_initializer_hook

def get_eval_inputs(batch_size, exercise_data):
	'''Return the input function to get the training data
	
		Args:
			batch_size (int): Batch size of training iterator that is returned by the input function
			excersize(mnist)_data (Object): Object holding the loaded mnist data
		
		Returns:
			(Input Function, IteratorInitializerHook)
			-Function that returns (features, labels) when called
			-Hook to initialize input iterator
	'''
	iterator_initializer_hook = IteratorInitializerHook()
	def eval_inputs():
		'''Returns test set as Operations
		
			Returns:
				(features, labels) Operations that iterate over the dataset on every evaluation
		'''	
		with tf.name_scope('Test_data'):
			#Get MINST data
			images = exercise_data[3]
			labels = exercise_data[2]
			#Define Placeholders
			images_placeholder = tf.placeholder(images.dtype, images.shape)
			labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
			#Build Dataset Iterator
			dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
			dataset = dataset.batch(batch_size)
			iterator = dataset.make_initializable_iterator()
			next_example, next_label = iterator.get_next()
			#Set runhook to initialize iterator
			iterator_initializer_hook.iterator_initializer_func = \
				lambda sess: sess.run(
					iterator.initializer,
					feed_dict={images_placeholder: images,
								labels_placeholder: labels})
			return next_example, next_label
		
	return eval_inputs, iterator_initializer_hook

def get_test_inputs(batch_size, exercise_data):
	'''Return the input function to get the training data
	
		Args:
			batch_size (int): Batch size of training iterator that is returned by the input function
			excersize(mnist)_data (Object): Object holding the loaded mnist data
		
		Returns:
			(Input Function, IteratorInitializerHook)
			-Function that returns (features, labels) when called
			-Hook to initialize input iterator
	'''
	iterator_initializer_hook = IteratorInitializerHook()
	def test_inputs():
		'''Returns test set as Operations
		
			Returns:
				(features, labels) Operations that iterate over the dataset on every evaluation
		'''	
		with tf.name_scope('Test_data'):
			#Get MINST data
			images = exercise_data[5]
			labels = exercise_data[4]
			#Define Placeholders
			images_placeholder = tf.placeholder(images.dtype, images.shape)
			labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
			#Build Dataset Iterator
			dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
			dataset = dataset.batch(batch_size)
			iterator = dataset.make_initializable_iterator()
			next_example, next_label = iterator.get_next()
			#Set runhook to initialize iterator
			iterator_initializer_hook.iterator_initializer_func = \
				lambda sess: sess.run(
					iterator.initializer,
					feed_dict={images_placeholder: images,
								labels_placeholder: labels})
			return next_example, next_label
		
	return test_inputs, iterator_initializer_hook

if __name__ == "__main__":
	tf.app.run(
		main=run_experiment
		)