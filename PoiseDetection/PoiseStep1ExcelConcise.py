import openpyxl
import os
import math

dirname = os.path.realpath('.')
filename = dirname + '\\Summary of Poise Final.xlsx'
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

for i in range (0, len(dataA)):
	if ( i < 72):
		section = 0
		col = math.floor(i/24)
		row = i % 24

	elif (i >= 72 and i < 144):
		section = 1
		col = math.floor((i-72)/24)
		row = i % 24

	else:
		section = 2
		col = math.floor((i-144)/24)
		row = i % 24
	
	offset = i % 2
	offset = offset + 2*section
	if col == 0:
		if (row < 2):
			worksheetP ['D' + str(5+offset)] = dataA[i]
		elif (row >= 2 and row< 4):
			worksheetP ['D' + str(11+offset)] = dataA[i]
		elif (row >= 4 and row < 6):
			worksheetP ['D' + str(17+offset)] = dataA[i]
		elif (row >= 6 and row < 8):
			worksheetP ['D' + str(28+offset)] = dataA[i]
		elif (row >= 8 and row < 10):
			worksheetP ['D' + str(34+offset)] = dataA[i]
		elif (row >= 10 and row < 12):
			worksheetP ['D' + str(40+offset)] = dataA[i]
		elif (row >= 12 and row < 14):
			worksheetP ['D' + str(51+offset)] = dataA[i]
		elif (row >= 14 and row < 16):
			worksheetP ['D' + str(57+offset)] = dataA[i]
		elif (row>= 16 and row < 18):
			worksheetP ['D' + str(63+offset)] = dataA[i]
		elif (row >= 18 and row < 20):
			worksheetP ['D' + str(74+offset)] = dataA[i]
		elif (row >= 20 and row < 22):
			worksheetP ['D' + str(80+offset)] = dataA[i]
		elif (row >= 22 and row < 24):
			worksheetP ['D' + str(86+offset)] = dataA[i]
	elif col == 1:
		if (row < 2):
			worksheetP ['H' + str(5+offset)] = dataA[i]
		elif (row >= 2 and row< 4):
			worksheetP ['H' + str(11+offset)] = dataA[i]
		elif (row >= 4 and row < 6):
			worksheetP ['H' + str(17+offset)] = dataA[i]
		elif (row >= 6 and row < 8):
			worksheetP ['H' + str(28+offset)] = dataA[i]
		elif (row >= 8 and row < 10):
			worksheetP ['H' + str(34+offset)] = dataA[i]
		elif (row >= 10 and row < 12):
			worksheetP ['H' + str(40+offset)] = dataA[i]
		elif (row >= 12 and row < 14):
			worksheetP ['H' + str(51+offset)] = dataA[i]
		elif (row >= 14 and row < 16):
			worksheetP ['H' + str(57+offset)] = dataA[i]
		elif (row>= 16 and row < 18):
			worksheetP ['H' + str(63+offset)] = dataA[i]
		elif (row >= 18 and row < 20):
			worksheetP ['H' + str(74+offset)] = dataA[i]
		elif (row >= 20 and row < 22):
			worksheetP ['H' + str(80+offset)] = dataA[i]
		elif (row >= 22 and row < 24):
			worksheetP ['H' + str(86+offset)] = dataA[i]

	elif col == 2:
		if (row < 2):
			worksheetP ['L' + str(5+offset)] = dataA[i]
		elif (row >= 2 and row< 4):
			worksheetP ['L' + str(11+offset)] = dataA[i]
		elif (row >= 4 and row < 6):
			worksheetP ['L' + str(17+offset)] = dataA[i]
		elif (row >= 6 and row < 8):
			worksheetP ['L' + str(28+offset)] = dataA[i]
		elif (row >= 8 and row < 10):
			worksheetP ['L' + str(34+offset)] = dataA[i]
		elif (row >= 10 and row < 12):
			worksheetP ['L' + str(40+offset)] = dataA[i]
		elif (row >= 12 and row < 14):
			worksheetP ['L' + str(51+offset)] = dataA[i]
		elif (row >= 14 and row < 16):
			worksheetP ['L' + str(57+offset)] = dataA[i]
		elif (row>= 16 and row < 18):
			worksheetP ['L' + str(63+offset)] = dataA[i]
		elif (row >= 18 and row < 20):
			worksheetP ['L' + str(74+offset)] = dataA[i]
		elif (row >= 20 and row < 22):
			worksheetP ['L' + str(80+offset)] = dataA[i]
		elif (row >= 22 and row < 24):
			worksheetP ['L' + str(86+offset)] = dataA[i]
	
myworkbook.save(filename)