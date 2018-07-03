#Import Libraries
import math
import io

#to get rid of warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Predetermined selection of Bodyparts (CHANGE FOR REFINEMENT)
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

dirname = os.path.realpath('.')
filename = dirname + '\\PoiseDetection\\Data\\TestNumber.txt'

numberTestFiles = open(filename,"r")
numberTests = numberTestFiles.read()


def foundTaskX(dx,i):
		if ((dx[i-5] < 0) and (dx[i-4] < 0) and (dx[i-3] < 0) and (dx[i-2] < 0) and (dx[i-1] < 0) and (dx[i+5] > 0) and (dx[i+4] > 0) and (dx[i+3] > 0) and (dx[i+2] > 0) and (dx[i+1] > 0)):
			return True
		
		elif ((dx[i+5] < 0) and (dx[i+4] < 0) and (dx[i+3] < 0) and (dx[i+2] < 0) and (dx[i+1] < 0) and (dx[i-5] > 0) and (dx[i-4] > 0) and (dx[i-3] > 0) and (dx[i-2] > 0) and (dx[i-1] > 0)):
			return True
		
		else:
			return False

def foundTaskY(dy,i):
		if ((dy[i-5] < 0) and (dy[i-4] < 0) and (dy[i-3] < 0) and (dy[i-2] < 0) and (dy[i-1] < 0) and (dy[i+5] > 0) and (dy[i+4] > 0) and (dy[i+3] > 0) and (dy[i+2] > 0) and (dy[i+1] > 0)):
			return True
		
		elif ((dy[i+5] < 0) and (dy[i+4] < 0) and (dy[i+3] < 0) and (dy[i+2] < 0) and (dy[i+1] < 0) and (dy[i-5] > 0) and (dy[i-4] > 0) and (dy[i-3] > 0) and (dy[i-2] > 0) and (dy[i-1] > 0)):
			return True
		
		else:
			return False

def foundTaskZ(dz,i):
		if ((dz[i-5] < 0) and (dz[i-4] < 0) and (dz[i-3] < 0) and (dz[i-2] < 0) and (dz[i-1] < 0) and (dz[i+5] > 0) and (dz[i+4] > 0) and (dz[i+3] > 0) and (dz[i+2] > 0) and (dz[i+1] > 0)):
			return True
		
		elif ((dz[i+5] < 0) and (dz[i+4] < 0) and (dz[i+3] < 0) and (dz[i+2] < 0) and (dz[i+1] < 0) and (dz[i-5] > 0) and (dz[i-4] > 0) and (dz[i-3] > 0) and (dz[i-2] > 0) and (dz[i-1] > 0)):
			return True
		
		else:
			return False


def main(argv = None):
	for i in range(0, int(numberTests)):
			for j in range(0,27):
				
				x=[]
				y=[]
				z=[]

				dx=[]
				dy=[]
				dz=[]

				tx=[]
				ty=[]
				tz=[]

				ts=[]

				for line in open(dirname + "\\PoiseDetection\\Data\\test" + str(i)+"\\Position_" + file_names[j]):
					row = line.split(',')
					x.append(float(row[0]))
					y.append(float(row[1]))
					z.append(float(row[2]))
					ts.append(float(row[3]))

				velfile = open(dirname + "\\PoiseDetection\\Data\\test" + str(i)+ "\\Velocity_" + file_names[j], "w")
				for k in range(5,len(x)):
					dx.append((x[k] - x[k-5])/5)
					dy.append((y[k] - y[k-5])/5)
					dz.append((z[k] - z[k-5])/5)
					velfile.write(str(dx[k-5]) + "," + str(dy[k-5]) + "," + str(dz[k-5]) + "," + str(ts[k])+ '\n')
				velfile.close()

				taskfile = open(dirname + "\\PoiseDetection\\Data\\test" + str(i)+ "\\Task_" + file_names[j], "w")
				for l in range(5,len(dx)-5): #16?
					if foundTaskX(dx,l):
						tx.append(1)
					else:
						tx.append(0)

					if foundTaskY(dy,l):
						ty.append(1)
					else:
						ty.append(0)

					if foundTaskZ(dz,l):
						tz.append(1)
					else:
						tz.append(0)

					taskfile.write(str(tx[l-5]) + "," + str(ty[l-5]) + "," + str(tz[l-5]) + "," + str(ts[l]) + '\n')
				taskfile.close()
				
#needed in order to call main
if __name__ == '__main__':
	main()



			