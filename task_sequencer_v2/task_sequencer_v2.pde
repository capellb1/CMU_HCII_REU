/*
Base Code:
Thomas Sanchez Lengeling.
 http://codigogenerativo.com/

 KinectPV2, Kinect for Windows v2 library for processing

 Skeleton color map example.
 Skeleton (x,y) positions are mapped to match the color Frame
 X,Y,Z position recorded in Meters with the SpineShoulder Joint as 0,0
 Time recorded in ms - Data captured every 16-18ms

 *****NOTE: TASK DATA IS INCORRECT*****

 Action Detection Algorithm:
 CMU HCII REU Summer 2018
 PI: Dr. Sieworek
 Students:  Blake Capella & Deepak Subramanian
 */
 
 import KinectPV2.*;
 import controlP5.*;
 import java.util.Date;
 import java.text.DateFormat;
 import java.text.SimpleDateFormat;
 import java.io.*;
 import java.io.File;
 import java.io.FileOutputStream;
 import java.io.IOException;
 import java.util.Arrays;
 import org.gicentre.utils.stat.*;
 import java.awt.Frame;
 
 KinectPV2 kinect;
 ControlP5 cp5;
 //ControlFrame positionGraph;
 
 //Data Recording Variables
 String data_folder; //Path to store data -- defined below
 boolean recording = false;
 int time = 0;
 long start_time = 0;
 String trial = "0";
 DataPoint start_position;
 
 //Data Storage and Computation Variables
 ArrayList<ArrayList<DataPoint>> position;
 ArrayList<ArrayList<DataPoint>> velocity;// List of Lists -- List containing the data for each node/bodypart as a seperate list
 ArrayList<ArrayList<DataPoint>> task_record;
 ArrayList<Integer> joint_indexs;
 ArrayList<String> file_names;
 int offset = 0;
 
 //Graph Array Stroage
 float[] timestamps;
 float[] timestampsTask;
 ArrayList<float[]> xPosition;
 ArrayList<float[]> yPosition;
 ArrayList<float[]> zPosition;
 ArrayList<float[]> xVelocity;
 ArrayList<float[]> yVelocity;
 ArrayList<float[]> zVelocity;
 ArrayList<ArrayList<Float>> xTask;
 ArrayList<ArrayList<Float>> yTask;
 ArrayList<ArrayList<Float>> zTask;
 float[] xTasks;
 float[] yTasks;
 float[] zTasks;
 
 //GUI Variables
 Textfield labelField;
 Textfield bodyField;
 String labelName;
 String bodyName;
 int bodyNumber;
 
 //Graph Initialization
 XYChart scatterplotPositionX;
 XYChart scatterplotPositionY;
 XYChart scatterplotPositionZ;
 XYChart scatterplotVelocityX;
 XYChart scatterplotVelocityY;
 XYChart scatterplotVelocityZ;
 XYChart scatterplotTaskX;
 XYChart scatterplotTaskY;
 XYChart scatterplotTaskZ;
 
 class DataPoint {
  float x;
  float y;
  float z;
  long ts;
 
  public DataPoint(float x, float y, float z, long ts){
    this.x = x;
    this.y = y;
    this.z = z;
    this.ts = ts;
  }
}
 
void setup() {
  //GUI creation
   cp5 = new ControlP5(this);
  
  cp5.addButton("Start_Recording")
     .setValue(0)
     .setPosition(100,100)
     .setSize(200,19)
     ;
  cp5.addButton("Stop_Recording")
     .setValue(100)
     .setPosition(100,120)
     .setSize(200,19)
     ;
     
     
   labelField = cp5.addTextfield("")
     .setValue("")
     .setColor(color(255,200,0))
     .setPosition(100, 80)
     .setSize(200, 19)
     .setColorCursor(color(255,200,0))
     ;
     
   cp5.addLabel("label")
     .setPosition(45, 85)
     .setColor(color(255,0,0))
     .setValue("Enter label:")
     .setSize(50, 19)
     ;
 
    //Define data path -- CHANGE FOR YOUR USE
   //data_folder = "D:\\CMU\\kinect_data\\"; 
   //data_folder = "C:\\Users\\Deepak Subramanian\\Documents\\Internship\\HCII Research (2018)\\task_sequencer_v2\\Data\\"; 
   data_folder = "C:\\Users\\Admin\\BlakeDeepak\\DataCollection";

   
   //creation and populating arrays to record and store data
   joint_indexs = new ArrayList<Integer>(Arrays.asList(new Integer[]
      {KinectPV2.JointType_Head,
      KinectPV2.JointType_Neck,
      KinectPV2.JointType_SpineShoulder,
      KinectPV2.JointType_SpineMid,
      KinectPV2.JointType_SpineBase,
      KinectPV2.JointType_ShoulderRight,
      KinectPV2.JointType_ShoulderLeft,
      KinectPV2.JointType_HipRight,
      KinectPV2.JointType_HipLeft,
      KinectPV2.JointType_ElbowRight,
      KinectPV2.JointType_WristRight,
      KinectPV2.JointType_HandRight,
      KinectPV2.JointType_HandTipRight,
      KinectPV2.JointType_ThumbRight,
      KinectPV2.JointType_ElbowLeft,
      KinectPV2.JointType_WristLeft,
      KinectPV2.JointType_HandLeft,
      KinectPV2.JointType_HandTipLeft,
      KinectPV2.JointType_ThumbLeft,
      KinectPV2.JointType_HipRight,
      KinectPV2.JointType_KneeRight,
      KinectPV2.JointType_AnkleRight,
      KinectPV2.JointType_FootRight,
      KinectPV2.JointType_HipLeft,
      KinectPV2.JointType_KneeLeft,
      KinectPV2.JointType_AnkleLeft,
      KinectPV2.JointType_FootLeft
      }));
    
    //Creating empty list containing a seperate empty list for each body part located at data[0] = headlist, data[1] =necklist
    //note redundancy of task_record in preparation for future work and possibly ML applications 
    position = new ArrayList<ArrayList<DataPoint>>();
    velocity = new ArrayList<ArrayList<DataPoint>>();
    task_record = new ArrayList<ArrayList<DataPoint>>();
    xTask = new ArrayList<ArrayList<Float>>();
    yTask = new ArrayList<ArrayList<Float>>();
    zTask = new ArrayList<ArrayList<Float>>();
    
    for(int i = 0; i<joint_indexs.size();i++){
      position.add(new ArrayList<DataPoint>()); 
      velocity.add(new ArrayList<DataPoint>());
      task_record.add(new ArrayList<DataPoint>());
      xTask.add(new ArrayList<Float>());
      yTask.add(new ArrayList<Float>());
      zTask.add(new ArrayList<Float>());      
    }
    
    //create [[filenames]] that corresponds with each bodypart/single index value
    //index value for each body part is listed next to its file name
    file_names = new ArrayList<String>(Arrays.asList(new String[]
      {"Head.csv",         //0
      "Neck.csv",          //1
      "SpineShoulder.csv", //2
      "SpineMid.csv",      //3
      "SpineBase.csv",     //4
      "ShoulderRight.csv", //5
      "ShoulderLeft.csv",  //6
      "HipRight.csv",      //7
      "HipLeft.csv",       //8
      "ElbowRight.csv",    //9
      "WristRight.csv",    //10
      "HandRight.csv",     //11
      "HandTipRight.csv",  //12
      "ThumbRight.csv",    //13
      "ElbowLeft.csv",     //14
      "WristLeft.csv",     //15
      "HandLeft.csv",      //16
      "HandTipLeft.csv",   //17
      "ThumbLeft.csv",     //18
      "HipRight.csv",      //19
      "KneeRight.csv",     //20
      "AnkleRight.csv",    //21
      "FootRight.csv",     //22
      "HipLeft.csv",       //23
      "KneeLeft.csv",      //24
      "AnkleLeft.csv",     //25
      "FootLeft.csv"       //26
      }));  
      
    //defining the kinect object
    kinect = new KinectPV2(this); 
    kinect.enableSkeletonColorMap(true);
    kinect.enableColorImg(true);
    kinect.enableSkeleton3DMap(true);
    size(960,540,P3D);
    kinect.init();
    
 } //end setup
    
void draw() {
  background(0);
  image(kinect.getColorImage(), 0, 0, width, height);
  
  //for drawing skeleton
  ArrayList<KSkeleton> skeletonArray =  kinect.getSkeletonColorMap();
  
  //for getting x,y,z coordinates
  ArrayList<KSkeleton> skeletonArray3d =  kinect.getSkeleton3d();
 
  //loop through skeletons in the frame (should be only one)
  for (int i = 0; i < skeletonArray.size(); i++) {
    KSkeleton skeleton = (KSkeleton) skeletonArray.get(i);
    KSkeleton skeleton3d = (KSkeleton) skeletonArray3d.get(i);
    
    //Determines presence of a skeleton and grabs data
    if ((skeleton.isTracked()) && (time == 1)) {
      KJoint[] joints = skeleton.getJoints();
      KJoint[] joints3d = skeleton3d.getJoints();
      
      //keeps track of system time in Seconds
      long ts = System.currentTimeMillis(); 
      int storage_index = 0;
      
      for(Integer joint_index : joint_indexs){
        float x = joints3d[joint_index].getX();
        float y = joints3d[joint_index].getY();
        float z = joints3d[joint_index].getZ();
        DataPoint dp = new DataPoint(x,y,z,ts);
        
        //rotates through each of the joints and adds its data point for that frame
        position.get(storage_index).add(dp);
        
        //Derivative Calculations
        //5 frame delay in the calculations due to looking 5 frames in the past to calculate derivative
        //5 chosen to provide smoothest velocity function
        if(offset >= 5)
        {
          float Dx = (position.get(storage_index).get(offset).x - position.get(storage_index).get(offset-5).x)/5; //TODO: confirm division by change in index vs by change in time
          float Dy = (position.get(storage_index).get(offset).y - position.get(storage_index).get(offset-5).y)/5;
          float Dz =(position.get(storage_index).get(offset).z - position.get(storage_index).get(offset-5).z)/5;
          
          //create and store derivatives as a single datapoint in a list of list
          DataPoint v = new DataPoint(Dx,Dy,Dz,ts);
          velocity.get(storage_index).add(v);
        }// end derivative calculation
        
        //Check for Task Completion for each body part along each axis
        if(offset >=16) {    
           if(foundTaskx(storage_index)){
             xTask.get(storage_index).add(1.0);
             DataPoint T = new DataPoint(1,0,0,ts);
             task_record.get(storage_index).add(T);
            }
            else {
              xTask.get(storage_index).add(0.0);
              DataPoint T = new DataPoint(0,0,0,ts);
              task_record.get(storage_index).add(T);
            }   
           if(foundTasky(storage_index)){
             yTask.get(storage_index).add(1.0);
             DataPoint T = new DataPoint(0,1,0,ts);
             task_record.get(storage_index).add(T);
            }
            else {
              yTask.get(storage_index).add(0.0);
              DataPoint T = new DataPoint(0,0,0,ts);
              task_record.get(storage_index).add(T);
            } 
            if(foundTaskz(storage_index)){
              zTask.get(storage_index).add(1.0);
              DataPoint T = new DataPoint(0,0,1,ts);
             task_record.get(storage_index).add(T);
            }
            else {
              zTask.get(storage_index).add(0.0);
              DataPoint T = new DataPoint(0,0,0,ts);
              task_record.get(storage_index).add(T);
            }
         } 
         
         //populate first 16 points with 0 to match axis between graphs
         else { 
           xTask.get(storage_index).add(0.0);
           yTask.get(storage_index).add(0.0);
           zTask.get(storage_index).add(0.0);
         }
        storage_index +=1;     
    }//end for loop
   
     offset++;
     
     //Drawing
     color col  = skeleton.getIndexColor();
     fill(col);
     stroke(col);
     drawBody(joints);

     //draw different color for each hand state
     drawHandState(joints[KinectPV2.JointType_HandRight]);
     drawHandState(joints[KinectPV2.JointType_HandLeft]);
    }
  }
  fill(255, 0, 0);
  text(frameRate, 50, 50);
}//end draw()

// start_recording button controller
public void Start_Recording() {
  time = 1;
  println("Skeleton is being recorded.");
  labelName = labelField.getText().trim();
  //trial++;
  if(labelName.length() > 0) {
    recording = true;
  } else {
    System.out.println("ERROR: Enter a label before starting recording");
  }
}// end start recording

// stop_recording button controller
public void Stop_Recording() {
  time = 0;
  offset = 0;
  if(!recording){
    println("Press 'Start_Recording' button before stop recording button.");
  } else {
    println("Recorded data is being saved!");
    recording = false;
    
  
    //determine the start time for normaliziation
    start_time = position.get(0).get(0).ts;
    
    //determine starting position by checking mid spine for normalization
    start_position = position.get(2).get(0);
    
    //save data
    try {
      //Reading Test Number and Incrementing
      File trialNumberFile = new File(data_folder+"\\TestNumber.txt");
      BufferedReader br = new BufferedReader(new FileReader(trialNumberFile)); 
      trial = br.readLine();
      br.close();
      int trialN = Integer.valueOf(trial) + 1;
      PrintWriter testNumber = new PrintWriter(new FileWriter(trialNumberFile));
      testNumber.write(String.valueOf(trialN));
      testNumber.close();
      
      //boolean success = new File(data_folder+"\\test" + (trial-1)+"_"+Integer.toString(trial_num)).mkdirs();
      boolean success = new File(data_folder+"\\test" + trial).mkdirs();
      if(!success){
        println("Dir not created.");
      }
      
      PrintWriter pwlabel = new PrintWriter(new FileOutputStream(data_folder+"\\test"+trial+"\\label.csv"));
      pwlabel.println(labelName);
      pwlabel.close();
      
      //Size of data
      int sizeofData = position.get(0).size();

      //Initialize data arrays for graphing
       timestamps = new float[sizeofData];
       xPosition = new ArrayList<float[]>();
       yPosition = new ArrayList<float[]>();
       zPosition = new ArrayList<float[]>();
       xVelocity = new ArrayList<float[]>();
       yVelocity = new ArrayList<float[]>();
       zVelocity = new ArrayList<float[]>();
       
       //Creates the arrays for each bodypart
       for (Integer storage_index = 0; storage_index<joint_indexs.size(); storage_index++){
          xPosition.add(new float[sizeofData]);
          yPosition.add(new float[sizeofData]);
          zPosition.add(new float[sizeofData]);
          xVelocity.add(new float[sizeofData]);
          yVelocity.add(new float[sizeofData]);
          zVelocity.add(new float[sizeofData]);
        }//End for loop
      
      //store position and velocity data
      for(Integer storage_index = 0; storage_index<joint_indexs.size(); storage_index++){ //iterate through joint lists
        PrintWriter pw = new PrintWriter(new FileOutputStream(data_folder+ "\\test" + trial+"\\Position_"+file_names.get(storage_index)));
        PrintWriter pwderiv = new PrintWriter(new FileOutputStream(data_folder+"\\test" + trial+"\\Velocity_"+file_names.get(storage_index)));
        PrintWriter pwtask = new PrintWriter(new FileOutputStream(data_folder+ "\\test" + trial+"\\Task_"+file_names.get(storage_index)));
        ArrayList<DataPoint> joint_data = position.get(storage_index);
        ArrayList<DataPoint> joint_dataV = velocity.get(storage_index);
        ArrayList<DataPoint> joint_dataT = task_record.get(storage_index);
        int i = 0;
        for (DataPoint dataPoint : joint_data){
            timestamps[i] = i;
            
            pw.println((dataPoint.x-start_position.x)+","+(dataPoint.y-start_position.y)+","+(dataPoint.z-start_position.z)+","+(dataPoint.ts-start_time));
            xPosition.get(storage_index)[i] = dataPoint.x;
            yPosition.get(storage_index)[i] = dataPoint.y;
            zPosition.get(storage_index)[i] = dataPoint.z;
            i++;
        }//end for
        
        i = 0;
        for (DataPoint dataPoint : joint_dataV){
          pwderiv.println((dataPoint.x-start_position.x)+","+(dataPoint.y-start_position.y)+","+(dataPoint.z-start_position.z)+","+(dataPoint.ts-start_time));
          System.out.println(dataPoint.x-start_position.x);
          xVelocity.get(storage_index)[i] = dataPoint.x;
          yVelocity.get(storage_index)[i] = dataPoint.y;
          zVelocity.get(storage_index)[i] = dataPoint.z;
          i++;
        }//end for
        
        i = 0;
        for (DataPoint dataPoint : joint_dataT){
          pwtask.println((dataPoint.x-start_position.x)+","+(dataPoint.y-start_position.y)+","+(dataPoint.z-start_position.z)+","+(dataPoint.ts-start_time));
          i++;
        }//end for
        
        pw.close();
        pwderiv.close();
        pwtask.close();
       }//end for loop
        
       //positionGraph = new ControlFrame(this, 1400, 1000, "graph");
        
       //reset data arrays
       task_record = new ArrayList<ArrayList<DataPoint>>();
       position = new ArrayList<ArrayList<DataPoint>>();
       velocity = new ArrayList<ArrayList<DataPoint>>();
       for(int i = 0; i<joint_indexs.size();i++){
         position.add(new ArrayList<DataPoint>());
         velocity.add(new ArrayList<DataPoint>());
         task_record.add(new ArrayList<DataPoint>());
       }//end for loop
       
     } catch (IOException e){
     e.printStackTrace();
  }//end of catch
 }
}//end store_data        
 
 //Real time null velocity detection
 //Algorithm looks 5 frames in the past and future from the frame in question to ensure that the data made a significant cross over the zero velocity axis
 //isolated by axis
 //5 is midpoint where change occurs     
boolean foundTaskx(int bodypart){
       if(velocity.get(bodypart).get(offset-15).x > 0 && velocity.get(bodypart).get(offset-14).x > 0 && velocity.get(bodypart).get(offset-13).x > 0 && velocity.get(bodypart).get(offset-12).x > 0 && velocity.get(bodypart).get(offset-11).x > 0 && velocity.get(bodypart).get(offset-10).x < 0 && velocity.get(bodypart).get(offset-9).x < 0 && velocity.get(bodypart).get(offset-8).x < 0 && velocity.get(bodypart).get(offset-7).x < 0 && velocity.get(bodypart).get(offset-6).x < 0){
          System.out.println("X-Null Velocity Detected" + offset);
          return true;
        }
       else if(velocity.get(bodypart).get(offset-15).x < 0 && velocity.get(bodypart).get(offset-14).x < 0 && velocity.get(bodypart).get(offset-13).x < 0 && velocity.get(bodypart).get(offset-12).x < 0 && velocity.get(bodypart).get(offset-11).x < 0 && velocity.get(bodypart).get(offset-10).x > 0 && velocity.get(bodypart).get(offset-9).x > 0 && velocity.get(bodypart).get(offset-8).x > 0 && velocity.get(bodypart).get(offset-7).x > 0 && velocity.get(bodypart).get(offset-6).x > 0){
          System.out.println("X-Null Velocity Detected" + offset);
          return true;
        }
        else{
          return false;
        }
}//end foundtaskx
  
boolean foundTasky(int bodypart){
      if(velocity.get(bodypart).get(offset-15).y > 0 && velocity.get(bodypart).get(offset-14).y > 0 && velocity.get(bodypart).get(offset-13).y > 0 && velocity.get(bodypart).get(offset-12).y > 0 && velocity.get(bodypart).get(offset-11).y > 0 && velocity.get(bodypart).get(offset-10).y < 0 && velocity.get(bodypart).get(offset-9).y < 0 && velocity.get(bodypart).get(offset-8).y < 0 && velocity.get(bodypart).get(offset-7).y < 0 && velocity.get(bodypart).get(offset-6).y < 0){
          System.out.println("Y-Null Velocity Detected" + offset);
          return true;
        }
       else if(velocity.get(bodypart).get(offset-15).y < 0 && velocity.get(bodypart).get(offset-14).y < 0 && velocity.get(bodypart).get(offset-13).y < 0 && velocity.get(bodypart).get(offset-12).y < 0 && velocity.get(bodypart).get(offset-11).y < 0 && velocity.get(bodypart).get(offset-10).y > 0 && velocity.get(bodypart).get(offset-9).y > 0 && velocity.get(bodypart).get(offset-8).y > 0 && velocity.get(bodypart).get(offset-7).y > 0 && velocity.get(bodypart).get(offset-6).y > 0){
          System.out.println("Y-Null Velocity Detected" + offset);
          return true;
        }
       else{
          return false;
        }
}//end foundtasky

boolean foundTaskz(int bodypart){
      if(velocity.get(bodypart).get(offset-15).z > 0 && velocity.get(bodypart).get(offset-14).z > 0 && velocity.get(bodypart).get(offset-13).z > 0 && velocity.get(bodypart).get(offset-12).z > 0 && velocity.get(bodypart).get(offset-11).z > 0 && velocity.get(bodypart).get(offset-10).z < 0 && velocity.get(bodypart).get(offset-9).z < 0 && velocity.get(bodypart).get(offset-8).z < 0 && velocity.get(bodypart).get(offset-7).z < 0 && velocity.get(bodypart).get(offset-6).z < 0){
          System.out.println("Z-Null Velocity Detected" + offset);
          return true;
        }
       else if(velocity.get(bodypart).get(offset-15).z < 0 && velocity.get(bodypart).get(offset-14).z < 0 && velocity.get(bodypart).get(offset-13).z < 0 && velocity.get(bodypart).get(offset-12).z < 0 && velocity.get(bodypart).get(offset-12).z < 0 && velocity.get(bodypart).get(offset-10).z > 0 && velocity.get(bodypart).get(offset-9).z > 0 && velocity.get(bodypart).get(offset-8).z > 0 && velocity.get(bodypart).get(offset-7).z > 0 && velocity.get(bodypart).get(offset-6).z > 0){
          System.out.println("Z-Null Velocity Detected" + offset);
          return true;
        } 
       else{
          return false;
        }
}//end foundtask z

//DRAW BODY
void drawBody(KJoint[] joints) {
  drawBone(joints, KinectPV2.JointType_Head, KinectPV2.JointType_Neck);
  drawBone(joints, KinectPV2.JointType_Neck, KinectPV2.JointType_SpineShoulder);
  drawBone(joints, KinectPV2.JointType_SpineShoulder, KinectPV2.JointType_SpineMid);
  drawBone(joints, KinectPV2.JointType_SpineMid, KinectPV2.JointType_SpineBase);
  drawBone(joints, KinectPV2.JointType_SpineShoulder, KinectPV2.JointType_ShoulderRight);
  drawBone(joints, KinectPV2.JointType_SpineShoulder, KinectPV2.JointType_ShoulderLeft);
  drawBone(joints, KinectPV2.JointType_SpineBase, KinectPV2.JointType_HipRight);
  drawBone(joints, KinectPV2.JointType_SpineBase, KinectPV2.JointType_HipLeft);

  // Right Arm
  drawBone(joints, KinectPV2.JointType_ShoulderRight, KinectPV2.JointType_ElbowRight);
  drawBone(joints, KinectPV2.JointType_ElbowRight, KinectPV2.JointType_WristRight);
  drawBone(joints, KinectPV2.JointType_WristRight, KinectPV2.JointType_HandRight);
  drawBone(joints, KinectPV2.JointType_HandRight, KinectPV2.JointType_HandTipRight);
  drawBone(joints, KinectPV2.JointType_WristRight, KinectPV2.JointType_ThumbRight);

  // Left Arm
  drawBone(joints, KinectPV2.JointType_ShoulderLeft, KinectPV2.JointType_ElbowLeft);
  drawBone(joints, KinectPV2.JointType_ElbowLeft, KinectPV2.JointType_WristLeft);
  drawBone(joints, KinectPV2.JointType_WristLeft, KinectPV2.JointType_HandLeft);
  drawBone(joints, KinectPV2.JointType_HandLeft, KinectPV2.JointType_HandTipLeft);
  drawBone(joints, KinectPV2.JointType_WristLeft, KinectPV2.JointType_ThumbLeft);

  // Right Leg
  drawBone(joints, KinectPV2.JointType_HipRight, KinectPV2.JointType_KneeRight);
  drawBone(joints, KinectPV2.JointType_KneeRight, KinectPV2.JointType_AnkleRight);
  drawBone(joints, KinectPV2.JointType_AnkleRight, KinectPV2.JointType_FootRight);

  // Left Leg
  drawBone(joints, KinectPV2.JointType_HipLeft, KinectPV2.JointType_KneeLeft);
  drawBone(joints, KinectPV2.JointType_KneeLeft, KinectPV2.JointType_AnkleLeft);
  drawBone(joints, KinectPV2.JointType_AnkleLeft, KinectPV2.JointType_FootLeft);

  drawJoint(joints, KinectPV2.JointType_HandTipLeft);
  drawJoint(joints, KinectPV2.JointType_HandTipRight);
  drawJoint(joints, KinectPV2.JointType_FootLeft);
  drawJoint(joints, KinectPV2.JointType_FootRight);

  drawJoint(joints, KinectPV2.JointType_ThumbLeft);
  drawJoint(joints, KinectPV2.JointType_ThumbRight);

  drawJoint(joints, KinectPV2.JointType_Head);
}//end drawbody

//draw joint
void drawJoint(KJoint[] joints, int jointType) {
  pushMatrix();
  translate(joints[jointType].getX()/2, joints[jointType].getY()/2, joints[jointType].getZ()/2);
  ellipse(0, 0, 25, 25);
  popMatrix();
}// end draw joint

//draw bone
void drawBone(KJoint[] joints, int jointType1, int jointType2) {
  pushMatrix();
  translate(joints[jointType1].getX()/2, joints[jointType1].getY()/2, joints[jointType1].getZ()/2);
  ellipse(0, 0, 25, 25);
  popMatrix();
  line(joints[jointType1].getX()/2, joints[jointType1].getY()/2, joints[jointType1].getZ()/2, joints[jointType2].getX()/2, joints[jointType2].getY()/2, joints[jointType2].getZ()/2);
} //end draw bone

//draw hand state
void drawHandState(KJoint joint) {
  noStroke();
  handState(joint.getState());
  pushMatrix();
  translate(joint.getX()/2, joint.getY()/2, joint.getZ()/2);
  ellipse(0, 0, 25, 25);
  popMatrix();
}//end draw handstate

/*
Different hand state
 KinectPV2.HandState_Open
 KinectPV2.HandState_Closed
 KinectPV2.HandState_Lasso
 KinectPV2.HandState_NotTracked
 */
void handState(int handState) {
  switch(handState) {
  case KinectPV2.HandState_Open:
    fill(0, 255, 0);
    break;
  case KinectPV2.HandState_Closed:
    fill(255, 0, 0);
    break;
  case KinectPV2.HandState_Lasso:
    fill(0, 0, 255);
    break;
  case KinectPV2.HandState_NotTracked:
    fill(255, 255, 255);
    break;
  }  
}//end handstate

/*
//Creating a new frame object
class ControlFrame extends PApplet {

  int w, h;
  PApplet parent;
  ControlP5 cp5;
  
  //Constructor for creating a frame
  public ControlFrame(PApplet _parent, int _w, int _h, String _name) {
    super();   
    parent = _parent;
    w=_w;
    h=_h;
    PApplet.runSketch(new String[]{this.getClass().getName()}, this);
  }
  
  public void setup() {
    this.surface.setSize(w, h);
    cp5 = new ControlP5(this);
    //GUI Objects from Frame
    bodyField = cp5.addTextfield("")
                  .setValue("")
                  .setColor(color(255,200,0))
                  .setPosition(450, 25)
                  .setSize(200, 20)
                  .setColorCursor(color(255,200,0))
                  ;
                 
    cp5.addLabel("bodyPart")
       .setPosition(375, 35)
       .setColor(color(255,0,0))
       .setValue("Enter bodyPart:")
       .setSize(50, 20)
       ;
       
    cp5.addButton("find_Body_Part")
       .setValue(100)
       .setPosition(375,55)
       .setSize(200,19)
       ; 
    
    //General Graph Font
    textFont(createFont("Arial",11),11);
    
    //Scatterplot Setup for Graph X
    scatterplotPositionX = new XYChart(this); 
    scatterplotPositionX.showXAxis(true); 
    scatterplotPositionX.showYAxis(true); 
    scatterplotPositionX.setXFormat("###,###");
    scatterplotPositionX.setXAxisLabel("timestamp");
    scatterplotPositionX.setYAxisLabel("xcoordinates\n");
    scatterplotPositionX.setMinY(-1.5);
    scatterplotPositionX.setMaxY(1.5);
    scatterplotPositionX.setPointColour(color(180,50,50,100));
    scatterplotPositionX.setPointSize(1.5);
    
    //Scatterplot Setup for Graph Y
    scatterplotPositionY = new XYChart(this);
    
    scatterplotPositionY.showXAxis(true); 
    scatterplotPositionY.showYAxis(true); 
    scatterplotPositionY.setXFormat("###,###");
    scatterplotPositionY.setXAxisLabel("timestamp");
    scatterplotPositionY.setYAxisLabel("xcoordinates\n");
    scatterplotPositionY.setMinY(-1.5);
    scatterplotPositionY.setMaxY(1.5);
    scatterplotPositionY.setPointColour(color(180,50,50,100));
    scatterplotPositionY.setPointSize(1.5);
    
    //Scatterplot Setup for Graph Z
    scatterplotPositionZ = new XYChart(this);
 
    scatterplotPositionZ.showXAxis(true); 
    scatterplotPositionZ.showYAxis(true); 
    scatterplotPositionZ.setXFormat("###,###");
    scatterplotPositionZ.setXAxisLabel("timestamp");
    scatterplotPositionZ.setYAxisLabel("xcoordinates\n");
    scatterplotPositionZ.setMinY(-1.5);
    scatterplotPositionZ.setMaxY(1.5);
    scatterplotPositionZ.setPointColour(color(180,50,50,100));
    scatterplotPositionZ.setPointSize(1.5);
    
     //Scatterplot Setup for Graph X
    scatterplotVelocityX = new XYChart(this); 
    
    scatterplotVelocityX.showXAxis(true); 
    scatterplotVelocityX.showYAxis(true); 
    scatterplotVelocityX.setXFormat("###,###");
    scatterplotVelocityX.setXAxisLabel("timestamp");
    scatterplotVelocityX.setYAxisLabel("xcoordinates\n");
    scatterplotVelocityX.setMinY(-.1);
    scatterplotVelocityX.setMaxY(.1);
    scatterplotVelocityX.setPointColour(color(180,50,50,100));
    scatterplotVelocityX.setPointSize(1.5);
    
    //Scatterplot Setup for Graph Y
    scatterplotVelocityY = new XYChart(this);
    
    scatterplotVelocityY.showXAxis(true); 
    scatterplotVelocityY.showYAxis(true); 
    scatterplotVelocityY.setXFormat("###,###");
    scatterplotVelocityY.setXAxisLabel("timestamp");
    scatterplotVelocityY.setYAxisLabel("xcoordinates\n");
    scatterplotVelocityY.setMinY(-.1);
    scatterplotVelocityY.setMaxY(.1);
    scatterplotVelocityY.setPointColour(color(180,50,50,100));
    scatterplotVelocityY.setPointSize(1.5);
    
    //Scatterplot Setup for Graph Z
    scatterplotVelocityZ = new XYChart(this);
 
    scatterplotVelocityZ.showXAxis(true); 
    scatterplotVelocityZ.showYAxis(true); 
    scatterplotVelocityZ.setXFormat("###,###");
    scatterplotVelocityZ.setXAxisLabel("timestamp");
    scatterplotVelocityZ.setYAxisLabel("xcoordinates");
    scatterplotVelocityZ.setMinY(-.1);
    scatterplotVelocityZ.setMaxY(.1);
    scatterplotVelocityZ.setPointColour(color(180,50,50,100));
    scatterplotVelocityZ.setPointSize(1.5);
    
    //Scatterplot Setup for Graph X
    scatterplotTaskX = new XYChart(this);
 
    scatterplotTaskX.showXAxis(true); 
    scatterplotTaskX.showYAxis(true); 
    scatterplotTaskX.setXFormat("###,###");
    scatterplotTaskX.setXAxisLabel("timestamp");
    scatterplotTaskX.setYAxisLabel("xFound");
    scatterplotTaskX.setMinY(0);
    scatterplotTaskX.setMaxY(1);
    scatterplotTaskX.setPointColour(color(180,180,50,180));
    scatterplotTaskX.setPointSize(4);
    
    //Scatterplot Setup for Graph Y
    scatterplotTaskY = new XYChart(this);
 
    scatterplotTaskY.showXAxis(true); 
    scatterplotTaskY.showYAxis(true); 
    scatterplotTaskY.setXFormat("###,###");
    scatterplotTaskY.setXAxisLabel("timestamp");
    scatterplotTaskY.setYAxisLabel("yFound");
    scatterplotTaskY.setMinY(0);
    scatterplotTaskY.setMaxY(1);
    scatterplotTaskY.setPointColour(color(180,180,50,180));
    scatterplotTaskY.setPointSize(4);
    
    //Scatterplot Setup for Graph Z
    scatterplotTaskZ = new XYChart(this);
 
    scatterplotTaskZ.showXAxis(true); 
    scatterplotTaskZ.showYAxis(true); 
    scatterplotTaskZ.setXFormat("###,###");
    scatterplotTaskZ.setXAxisLabel("timestamp");
    scatterplotTaskZ.setYAxisLabel("zFound");
    scatterplotTaskZ.setMinY(0);
    scatterplotTaskZ.setMaxY(1);
    scatterplotTaskZ.setPointColour(color(180,180,50,180));
    scatterplotTaskZ.setPointSize(4);
  } //End Setup() for Frame
  
  void draw(){
      //Creating the Graph Titles
      text("Position over Time", 200, 50);
      text("X-Coordinates over Time", 200, 100);
      text("Y-Coordinates over Time", 200, 400);
      text("Z-Coordinates over Time", 200, 700);
      
      text("Velocity over Time", 700, 50);
      text("X-Coordinates over Time", 700, 100);
      text("Y-Coordinates over Time", 700, 400);
      text("Z-Coordinates over Time", 700, 700);
      
      text("Velocity over Time", 1100, 50);
      text("X-Coordinates over Time", 1100, 100);
      text("Y-Coordinates over Time", 1100, 400);
      text("Z-Coordinates over Time", 1100, 700);
      
      //Drawing the Graphs
      scatterplotPositionX.draw(20,100,450,250);
      scatterplotPositionY.draw(20,400,450,250);
      scatterplotPositionZ.draw(20,700,450,250);
      
      scatterplotVelocityX.draw(470,100,450,250);
      scatterplotVelocityY.draw(470,400,450,250);
      scatterplotVelocityZ.draw(470,700,450,250);
      
      scatterplotTaskX.draw(920,100,450,250);
      scatterplotTaskY.draw(920,400,450,250);
      scatterplotTaskZ.draw(920,700,450,250);
  }//End Draw() for Frame
  
  public void find_Body_Part() {
      //Reads input field
      bodyName = bodyField.getText().trim();
      //Checks that text is entered
      if(bodyName.length() > 0) {
        bodyNumber = Integer.parseInt(bodyName);
      } 
      else {
        System.out.println("Please enter Body Part");
        bodyNumber = 0;
      }  
      
      //Creating Task Graph Data
      xTasks = new float[xTask.get(bodyNumber).size()];
      yTasks = new float[yTask.get(bodyNumber).size()];
      zTasks = new float[zTask.get(bodyNumber).size()];
      timestampsTask = new float[xTask.get(bodyNumber).size()];
      
      for (int i = 0; i < xTask.get(bodyNumber).size(); i++){
        xTasks[i] = xTask.get(bodyNumber).get(i);
        yTasks[i] = yTask.get(bodyNumber).get(i);
        zTasks[i] = zTask.get(bodyNumber).get(i);
        timestampsTask[i] = i;
      }
      
      //System.out.println(xTasks.length);
      //System.out.println(timestampsTask.length);
      
      //Resets Graph
      background(0);
      //Sets respective data
      scatterplotPositionX.setData(timestamps,xPosition.get(bodyNumber));
      scatterplotPositionY.setData(timestamps,yPosition.get(bodyNumber));
      scatterplotPositionZ.setData(timestamps,zPosition.get(bodyNumber));
      
      scatterplotVelocityX.setData(timestamps,xVelocity.get(bodyNumber));
      scatterplotVelocityY.setData(timestamps,yVelocity.get(bodyNumber));
      scatterplotVelocityZ.setData(timestamps,zVelocity.get(bodyNumber));
      
      scatterplotTaskX.setData(timestampsTask,xTasks);
      scatterplotTaskY.setData(timestampsTask,yTasks);
      scatterplotTaskZ.setData(timestampsTask,zTasks);
      
      text(file_names.get(bodyNumber), 585, 65);
    }//End find_Body_Part()
}//Ends definition for Frame
*/