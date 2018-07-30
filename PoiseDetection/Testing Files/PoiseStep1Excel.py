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
	col = math.floor(i / 72)
	row = i%72
	if col == 0:
		if (row < 18):
			worksheetP ['D' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['D' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['D' + str(15+row)] = dataA[i]
		else:
			worksheetP ['D' + str(20+row)] = dataA[i]
	elif col == 1:
		if (row < 18):
			worksheetP ['E' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['E' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['E' + str(15+row)] = dataA[i]
		else:
			worksheetP ['E' + str(20+row)] = dataA[i]

	elif col == 2:
		if (row < 18):
			worksheetP ['F' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['F' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['F' + str(15+row)] = dataA[i]
		else:
			worksheetP ['F' + str(20+row)] = dataA[i]
	elif col == 3:
		if (row < 18):
			worksheetP ['H' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['H' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['H' + str(15+row)] = dataA[i]
		else:
			worksheetP ['H' + str(20+row)] = dataA[i]

	elif col == 4:
		if (row < 18):
			worksheetP ['I' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['I' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['I' + str(15+row)] = dataA[i]
		else:
			worksheetP ['I' + str(20+row)] = dataA[i]
	elif col == 5:
		if (row < 18):
			worksheetP ['J' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['J' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['J' + str(15+row)] = dataA[i]
		else:
			worksheetP ['J' + str(20+row)] = dataA[i]
	elif col == 6:
		if (row < 18):
			worksheetP ['L' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['L' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['L' + str(15+row)] = dataA[i]
		else:
			worksheetP ['L' + str(20+row)] = dataA[i]
	elif col == 8:
		if (row < 18):
			worksheetP ['M' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['M' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['M' + str(15+row)] = dataA[i]
		else:
			worksheetP ['M' + str(20+row)] = dataA[i]
	elif col == 9:
		if (row < 18):
			worksheetP ['N' + str(5+row)] = dataA[i]
		elif (row >= 18 and row < 36):
			worksheetP ['N' + str(10+row)] = dataA[i]
		elif (row >= 36 and row < 54):
			worksheetP ['N' + str(15+row)] = dataA[i]
		else:
			worksheetP ['N' + str(20+row)] = dataA[i]
myworkbook.save(filename)