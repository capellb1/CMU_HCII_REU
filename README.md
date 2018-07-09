# Carnegie Mellon University- Human Computer Interaction Insitute

## Research Experience for Undergraduates (REU) Summer 2018

### Students: Blake Capella & Deepak Subramanian

### PI: Dr. Daniel Siewiorek

### Assisting Professors: Dr. Asim Smailagic & Dr. Roberta Klatzky

Includes code related to the creation of a cognitive assistant using the Microsoft Kinect 2 through real time data processing and machine learning.

The source of the data being used to train the fully connected neural net can be toggled between the natural data (Position) and synthetic/calculated features (Position or Task). This is controlled by the --source flag.

Many flags might not be used in everu file, they were included for consistency between the multiple training files:

	poise_detector_mk*.py
	poise_detector_batch_mk*.py
	exercise_detection_mk*.py

Unless otherwise stated, assume highest number to be the most current/advanced file

You must have at least 5 files in order to train a model

Assumes that you are reading from a data library constructed by the task_sequencer_v2.pde file
If not, organize your data as follows:

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

Flags:

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