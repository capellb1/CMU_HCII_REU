import openpyxl
import os
import math

dirname = os.path.realpath('.')
filename = dirname + '\\Summary of Poise Steps 2 and 3.xlsx'
data = dirname + '\\Models&Results\\totalResults.txt'
myworkbook = openpyxl.load_workbook(filename)

worksheetP = myworkbook['Final Poise Position']

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
	if ( i < 72):
		col = math.floor(i / 24)
		row = i % 24
		if ((i < 18) or (i >= 24 and i < 42) or (i >= 48 and i < 66)):
			if col == 0:
				worksheetP ['E' + str(4+row)] = dataA[i]
			elif col == 1:
				worksheetP ['F' + str(4+row)] = dataA[i]
			else:
				worksheetP ['G' + str(4+row)] = dataA[i]
		else:
			if col == 0:
				worksheetP ['E' + str(7+row)] = dataA[i]
			elif col == 1:
				worksheetP ['F' + str(7+row)] = dataA[i]
			else:
				worksheetP ['G' + str(7+row)] = dataA[i]			
	elif ( i >= 72 and i < 144):
		col = math.floor(i / 24)
		row = i % 24
		if ((i >= 72 and i < 90) or (i >= 96 and i < 114) or (i >= 120 and i < 138)):
			if col == 0:
				worksheetP ['I' + str(4+row)] = dataA[i]
			elif col == 1:
				worksheetP ['J' + str(4+row)] = dataA[i]
			else:
				worksheetP ['K' + str(4+row)] = dataA[i]
		else:
			if col == 0:
				worksheetP ['I' + str(16+row)] = dataA[i]
			elif col == 1:
				worksheetP ['J' + str(16+row)] = dataA[i]
			else:
				worksheetP ['K' + str(16+row)] = dataA[i]
	else:
		col = math.floor(i / 36)
		row = i % 18
		if ((i >= 144 and i < 162) or (i >= 168 and i < 186) or (i >= 192 and i < 210)):
			if col == 0:
				worksheetP ['M' + str(4+row)] = dataA[i]
			elif col == 1:
				worksheetP ['N' + str(4+row)] = dataA[i]
			else:
				worksheetP ['O' + str(4+row)] = dataA[i]
		else:
			if col == 0:
				worksheetP ['M' + str(7+row)] = dataA[i]
			elif col == 1:
				worksheetP ['N' + str(7+row)] = dataA[i]
			else:
				worksheetP ['O' + str(7+row)] = dataA[i]		
myworkbook.save(filename)
 