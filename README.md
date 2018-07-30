# Carnegie Mellon University- Human Computer Interaction Insitute

## Research Experience for Undergraduates (REU) Summer 2018

#### Students: Blake Capella & Deepak Subramanian

#### PI: Dr. Daniel Siewiorek

#### Assisting Professors: Dr. Asim Smailagic & Dr. Roberta Klatzky

### Models/Training

*__WARNING:__* In cleaning up the file tree in order to reduce the number of duplicate files, the pathing has changed. This change will primarily be seen in the process of reading the data and storing results

This project includes code related to the creation of a cognitive assistant using the _Microsoft Kinect 2_ through real time data processing and machine learning.

The source of the data being used to train the fully connected neural net is the X, Y, and Z position of 25 different joints:

	Head   
	Neck    
	SpineShoulder 
	SpineMid
	SpineBase    
	ShoulderRight 
	ShoulderLeft  
	HipRight
	HipLeft 
	ElbowRight    
	WristRight    
	HandRigh     
	HandTipRight  
	ThumbRight   
	ElbowLeft     
	WristLeft     
	HandLeft    
	HandTipLeft  
	ThumbLeft    
	KneeRight    
	AnkleRight   
	FootRigh     
	KneeLeft
	AnkleLeft     
	FootLeft

 In the experiment, the team specifically explored the use of several tools in order to determine the best possible performance 
 for a task detector and classifier:
 	
 - Uniformed Refinement: Predefining the joints to include
 
 - Tailored Refinement: Dynamically choosing the joints to use by examining the activity of each joint
 
 - Transformed Features: The input data can be toggled between the natural data (Position) and any combination of synthetic						 calculated features (Position, Task, Velocity) for each joint.
 
 - Downsampling (50%)
 
 - Dynamic Scope Definition: Change what portion of the activity the model is evaluating based on characterestics taken from
 							the frame by frame analysis

Many flags might not be used in every file, they were included for consistency between the multiple training files:

	poise_detector_mk*.py
	exercise_detection_mk*.py

Unless otherwise stated, assume highest number to be the most current/advanced file. Older files will be stored in the ARCHIVED folder.

The other files included in the github are for various different utilities. The most important of these being:

	reloadModel.py

This file restores a saved model and uses it to predict/test on new data. The poise variant of this file also provides a histogram
that illustrates the cumulative frame by frame accuracy on each exercise. In order to reload a model, add the same flags you used to train the model to the command line input for reloadModel.py

Other tools related to data preprocessing and error correction can be found in the -'Tools'- folder in the home directory. Each file contains a header with basic information on the file's function. As they are currently written, the files must be run in the same directory that holds the data you are using. The name of this folder can be easily changed by changing the "DATA_FOLDER" constant.

With the current setup, you must have at least 5 files in order to train a model. This is due to the training, testing split of 70 and 30. In some of the archived files you will also see a validation data split, however this was discarded later in the project.

Assumes that you are reading from a data library constructed by the task_sequencer_v2.pde file. An equally distributed subsample of this data that includes 10 examples of each exercise exists in the 'DataCollectionSample' folder. This is the primary set of data the team used in testing models.

If you are looking to use your own data, organize your data as follows:

	Data
		test0
	 		Position_Head.csv (organized by x,y,z,ts)
			Position_Neck.csv
			...
			Velocity_Head.csv
			...	
			Task_Head.csv
		test1
		test2
	 	...
		TestNumber.txt (stores total number of examples/identified actions)
	TRAINING FILE

Otherwise, organize code as you see fit

Unless stated, the input for flags can be any keyboard input

__Flags:__

	batch_size 
		number of randomly sampled images from the training set
			DEFAULT: 1000(poise) or 10(exercise)
	learning_rate
		how quickly the model progresses along the loss curve during optimization
			DEFAULT: 0.001
	epochs
		number of passes over the training data
			DEFAULT: 100
	regularization_rate
		Strength of regularization
			DEFAULT: 0.01
	regularization
		the regularization function used in cost calcuations, useful in preventing overfitting
			DEFAULT: None
			Options: None, L1, L2
	activation
		the activation function to use in the layers
			DEFAULT: None
			Options: None, Sigmoid, Tanh, Relu
	label
		the label name for where data files are saved for the model being trained
			DEFAULT: Test1
	position
		Whether or not to include the position data in the training of the model. 
		"--position" asserts a true
			DEFAULT: False
	velocity
		Whether or not to include the velocity data in the training of the model. 
		"--velocity" asserts a true
			DEFAULT: False
	mode
		Used for reloadModel.py. Determines how the statistics are determined
			OPTIONS: Test, Predict
	verbose
		Increases the amount of data written to the results file
			DEFAULT: False
	refinement
		Determines the method of refinement
			DEFAULT: None
			OPTIONS: Uniform, Tailored, None
	refinement rate:
		Determines the number of joints to check
		0 = 25 joints
		25 = 19 joints (minus hands/feet)
		50 = 13 joints (head, shoulder, elbow, wrist, hip, knee, ankle)
		75 = 6 joints (shoulder, wrist, ankle)
	arch
		specifies the architecture to use
		method1 = 60x60
		method2 = 40x40x40
		method3 = 30x30x30x30
		method4 = 24x24x24x24x24
			DEFAULT: method1
			Options: method1, method2, method3, method4

### Dataset

Data collected by a __Kinect V2__ as a set of X, Y, Z coordinates at 60fps. The program used to record this data was adapted from _Thomas Sanchez Langeling’s_ skeleton recording code.  The file was set to record data for each body part as a separate file, repeated for each exercise. These coordinates were chosen to have an origin centered at the subject’s upper chest. Data collection was standardized to the following conditions:

- Kinect placed at the height of 2ft and 3in
- Subject consistently positioned 6.5 feet away from the camera with their chests facing the camera
- Followed Script and Tutorial Video

Data was collected from the following population:

- Adults ages 18-21
- Girls: 4
- Guys: 5

The following types of pre-processing occurred at the time of data collection.
- Velocity Data: Calculated using a discrete derivative equation with a spacing of 5 samples 
	5 frames chosen to reduce sensitivity of the velocity function
  - v[n]=(x[n]-x[n-5])/5
- Task Data: Built on the philosophy that zero velocity points will mark the end of an action
	At the point under consideration, if there are 5 points ahead and 5 points behind that are opposite signs, a binary value of 1 is recorded. Otherwise a zero is recorded
  - Occurs for all body parts and all axis individually
