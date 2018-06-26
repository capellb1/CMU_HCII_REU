# CMU_HCII_REU
Carnegie Mellon University- Human Computer Interaction Insitute
Research Experience for Undergraduates

Students: Blake Capella & Deepak Subramanian

PI: Dr. Daniel Siewiorek
Assisting Professors: Dr. Asim Smailagic & Dr. Roberta Klatzky

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
	TRAINING FILE

Otherwise, organize code as you see fit

Unless stated, the input for flags can be any keyboard input
Flags:
	--batch_size 
		number of randomly sampled images from the training set
			DEFAULT: 1000(poise) or 10(exercise)
	--learning_rate
		how quickly the model progresses along the loss curve during optimization
			DEFAULT: 0.001
	--epochs
		number of passes over the training data
			DEFAULT: 100
	--regularization_rate
		Strength of regularization
			DEFAULT: 0.01
	--regularization
		the regularization function used in cost calcuations, useful in preventing overfitting
			DEFAULT: None
			Options: None, L1, L2
	--activation
		the activation function to use in the layers
			DEFAULT: None
			Options: None, Sigmoid, Tanh, Relu
	--label
		the label name for where data files are saved for the model being trained
			DEFAULT: Test1
	--frames
		Number of frames to be analyzed at a time
		Only used for poise_detector_batch.mk*.py
			DEFAULT: 5
	--source
		What files to draw data from
			DEFAULT: Position
			Options: Position, Task, Velocity
	--arch
		specifies the architecture to use
		method1 = 30x30
		method2 = 10x10x10
		method3 = 15x15
			DEFAULT: method1
			Options: method1, method2, method3