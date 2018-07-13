import openpyxl
import os
import math

dirname = os.path.realpath('.')
filename = dirname + '\\Summary of Exercises Final.xlsx'
data = dirname + '\\Models&Results\\totalResults.txt'
myworkbook = openpyxl.load_workbook(filename)

worksheetP = myworkbook['Step 1']

f=open(data)
lines=f.readlines()

dataA = []
for line in lines:
	newLine = line.split(':')
	newLine = newLine[1].split('\n')
	dataA.append(float((newLine[0])))

for j in range (0, len(dataA)):
	col = math.floor(j / 24)
	row = j%2 + 4 
	i = j % 24

	if col == 0:
		if (i < 2):
			worksheetP ['D' + str(5+row)] = dataA[i]
		elif (i >= 2 and i< 4):
			worksheetP ['D' + str(11+row)] = dataA[i]
		elif (i >= 4 and i < 6):
			worksheetP ['D' + str(17+row)] = dataA[i]
		elif (i>= 6 and i < 8):
			worksheetP ['D' + str(28+row)] = dataA[i]
		elif (i >= 8 and i < 10):
			worksheetP ['D' + str(34+row)] = dataA[i]
		elif (i >= 10 and i < 12):
			worksheetP ['D' + str(40+row)] = dataA[i]
		elif (i >= 12 and i < 14):
			worksheetP ['D' + str(51+row)] = dataA[i]
		elif (i >= 14 and i < 16):
			worksheetP ['D' + str(57+row)] = dataA[i]
		elif (i>= 16 and i < 18):
			worksheetP ['D' + str(63+row)] = dataA[i]
		elif (i >= 18 and i < 20):
			worksheetP ['D' + str(74+row)] = dataA[i]
		elif (i >= 20 and i < 22):
			worksheetP ['D' + str(80+row)] = dataA[i]
		elif (i >= 22 and i < 24):
			worksheetP ['D' + str(86+row)] = dataA[i]
	elif col == 1:
		if (i < 2):
			worksheetP ['H' + str(5+row)] = dataA[i]
		elif (i >= 2 and i< 4):
			worksheetP ['H' + str(11+row)] = dataA[i]
		elif (i >= 4 and i < 6):
			worksheetP ['H' + str(17+row)] = dataA[i]
		elif (i>= 6 and i < 8):
			worksheetP ['H' + str(28+row)] = dataA[i]
		elif (i >= 8 and i < 10):
			worksheetP ['H' + str(34+row)] = dataA[i]
		elif (i >= 10 and i < 12):
			worksheetP ['H' + str(40+row)] = dataA[i]
		elif (i >= 12 and i < 14):
			worksheetP ['H' + str(51+row)] = dataA[i]
		elif (i >= 14 and i < 16):
			worksheetP ['H' + str(57+row)] = dataA[i]
		elif (i>= 16 and i < 18):
			worksheetP ['H' + str(63+row)] = dataA[i]
		elif (i >= 18 and i < 20):
			worksheetP ['H' + str(74+row)] = dataA[i]
		elif (i >= 20 and i < 22):
			worksheetP ['H' + str(80+row)] = dataA[i]
		elif (i >= 22 and i < 24):
			worksheetP ['H' + str(86+row)] = dataA[i]

	elif col == 2:
		if (i < 2):
			worksheetP ['L' + str(5+row)] = dataA[i]
		elif (i >= 2 and i< 4):
			worksheetP ['L' + str(11+row)] = dataA[i]
		elif (i >= 4 and i < 6):
			worksheetP ['L' + str(17+row)] = dataA[i]
		elif (i>= 6 and i < 8):
			worksheetP ['L' + str(28+row)] = dataA[i]
		elif (i >= 8 and i < 10):
			worksheetP ['L' + str(34+row)] = dataA[i]
		elif (i >= 10 and i < 12):
			worksheetP ['L' + str(40+row)] = dataA[i]
		elif (i >= 12 and i < 14):
			worksheetP ['L' + str(51+row)] = dataA[i]
		elif (i >= 14 and i < 16):
			worksheetP ['L' + str(57+row)] = dataA[i]
		elif (i>= 16 and i < 18):
			worksheetP ['L' + str(63+row)] = dataA[i]
		elif (i >= 18 and i < 20):
			worksheetP ['L' + str(74+row)] = dataA[i]
		elif (i >= 20 and i < 22):
			worksheetP ['L' + str(80+row)] = dataA[i]
		elif (i >= 22 and i < 24):
			worksheetP ['L' + str(86+row)] = dataA[i]
	
myworkbook.save(filename)