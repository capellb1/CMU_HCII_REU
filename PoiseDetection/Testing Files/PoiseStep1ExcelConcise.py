import openpyxl
import os
import math

dirname = os.path.realpath('.')
filename = dirname + '\\Summary of Poise Final.xlsx'
data = dirname + '\\Models&Results\\totalResults.txt'
myworkbook = openpyxl.load_workbook(filename)

worksheetP = myworkbook['Preliminary Step']

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
			worksheetP ['F' + str(5+offset)] = dataA[i]
		elif (row >= 2 and row< 4):
			worksheetP ['F' + str(11+offset)] = dataA[i]
		elif (row >= 4 and row < 6):
			worksheetP ['F' + str(17+offset)] = dataA[i]
		elif (row >= 6 and row < 8):
			worksheetP ['F' + str(28+offset)] = dataA[i]
		elif (row >= 8 and row < 10):
			worksheetP ['F' + str(34+offset)] = dataA[i]
		elif (row >= 10 and row < 12):
			worksheetP ['F' + str(40+offset)] = dataA[i]
		elif (row >= 12 and row < 14):
			worksheetP ['F' + str(51+offset)] = dataA[i]
		elif (row >= 14 and row < 16):
			worksheetP ['F' + str(57+offset)] = dataA[i]
		elif (row>= 16 and row < 18):
			worksheetP ['F' + str(63+offset)] = dataA[i]
		elif (row >= 18 and row < 20):
			worksheetP ['F' + str(74+offset)] = dataA[i]
		elif (row >= 20 and row < 22):
			worksheetP ['F' + str(80+offset)] = dataA[i]
		elif (row >= 22 and row < 24):
			worksheetP ['F' + str(86+offset)] = dataA[i]
	elif col == 1:
		if (row < 2):
			worksheetP ['J' + str(5+offset)] = dataA[i]
		elif (row >= 2 and row< 4):
			worksheetP ['J' + str(11+offset)] = dataA[i]
		elif (row >= 4 and row < 6):
			worksheetP ['J' + str(17+offset)] = dataA[i]
		elif (row >= 6 and row < 8):
			worksheetP ['J' + str(28+offset)] = dataA[i]
		elif (row >= 8 and row < 10):
			worksheetP ['J' + str(34+offset)] = dataA[i]
		elif (row >= 10 and row < 12):
			worksheetP ['J' + str(40+offset)] = dataA[i]
		elif (row >= 12 and row < 14):
			worksheetP ['J' + str(51+offset)] = dataA[i]
		elif (row >= 14 and row < 16):
			worksheetP ['J' + str(57+offset)] = dataA[i]
		elif (row>= 16 and row < 18):
			worksheetP ['J' + str(63+offset)] = dataA[i]
		elif (row >= 18 and row < 20):
			worksheetP ['J' + str(74+offset)] = dataA[i]
		elif (row >= 20 and row < 22):
			worksheetP ['J' + str(80+offset)] = dataA[i]
		elif (row >= 22 and row < 24):
			worksheetP ['J' + str(86+offset)] = dataA[i]

	elif col == 2:
		if (row < 2):
			worksheetP ['N' + str(5+offset)] = dataA[i]
		elif (row >= 2 and row< 4):
			worksheetP ['N' + str(11+offset)] = dataA[i]
		elif (row >= 4 and row < 6):
			worksheetP ['N' + str(17+offset)] = dataA[i]
		elif (row >= 6 and row < 8):
			worksheetP ['N' + str(28+offset)] = dataA[i]
		elif (row >= 8 and row < 10):
			worksheetP ['N' + str(34+offset)] = dataA[i]
		elif (row >= 10 and row < 12):
			worksheetP ['N' + str(40+offset)] = dataA[i]
		elif (row >= 12 and row < 14):
			worksheetP ['N' + str(51+offset)] = dataA[i]
		elif (row >= 14 and row < 16):
			worksheetP ['N' + str(57+offset)] = dataA[i]
		elif (row>= 16 and row < 18):
			worksheetP ['N' + str(63+offset)] = dataA[i]
		elif (row >= 18 and row < 20):
			worksheetP ['N' + str(74+offset)] = dataA[i]
		elif (row >= 20 and row < 22):
			worksheetP ['N' + str(80+offset)] = dataA[i]
		elif (row >= 22 and row < 24):
			worksheetP ['N' + str(86+offset)] = dataA[i]
	
myworkbook.save(filename)