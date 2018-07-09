import openpyxl
import os
import math

dirname = os.path.realpath('.')
filename = dirname + '\\Summary of Poise Step 1.xlsx'
data = dirname + '\\Models&Results\\totalResults.txt'
myworkbook = openpyxl.load_workbook(filename)

worksheetP = myworkbook['Position']

f=open(data)
lines=f.readlines()

dataA = []
for line in lines:
	print (line)
	newLine = line.split(':')
	print(newLine)
	newLine = newLine[1].split('\n')
	dataA.append(float((newLine[0])))

print (dataA)
print (len(dataA))

for i in range (0, len(dataA)):
	col = math.floor(i / 18)
	row = i%18
	if col == 0:
		if (row < 6):
			worksheetP ['D' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['D' + str(7+row)] = dataA[i]
		else:
			worksheetP ['D' + str(10+row)] = dataA[i]
	elif col == 1:
		if (row < 6):
			worksheetP ['E' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['E' + str(7+row)] = dataA[i]
		else:
			worksheetP ['E' + str(10+row)] = dataA[i]

	elif col == 2:
		if (row < 6):
			worksheetP ['F' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['F' + str(7+row)] = dataA[i]
		else:
			worksheetP ['F' + str(10+row)] = dataA[i]

	elif col == 3:
		if (row < 6):
			worksheetP ['H' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['H' + str(7+row)] = dataA[i]
		else:
			worksheetP ['H' + str(10+row)] = dataA[i]

	elif col == 4:
		if (row < 6):
			worksheetP ['I' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['I' + str(7+row)] = dataA[i]
		else:
			worksheetP ['I' + str(10+row)] = dataA[i]
	elif col == 5:
		if (row < 6):
			worksheetP ['J' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['J' + str(7+row)] = dataA[i]
		else:
			worksheetP ['J' + str(10+row)] = dataA[i]

	elif col == 6:
		if (row < 6):
			worksheetP ['L' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['L' + str(7+row)] = dataA[i]
		else:
			worksheetP ['L' + str(10+row)] = dataA[i]

	elif col == 8:
		if (row < 6):
			worksheetP ['M' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['M' + str(7+row)] = dataA[i]
		else:
			worksheetP ['M' + str(10+row)] = dataA[i]

	elif col == 9:
		if (row < 6):
			worksheetP ['N' + str(4+row)] = dataA[i]
		elif (row >= 6 and row < 12):
			worksheetP ['N' + str(7+row)] = dataA[i]
		else:
			worksheetP ['N' + str(10+row)] = dataA[i]

myworkbook.save(filename)