import openpyxl
import os
import math

dirname = os.path.realpath('.')
filename = dirname + '\\Summary of Poise Steps 2 and 3.xlsx'
data = dirname + '\\Models&Results\\totalResults.txt'
myworkbook = openpyxl.load_workbook(filename)

worksheetP = myworkbook['Best Poise Position']

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
	col = math.floor(i / 12)
	row = i%12
	if ((i < 6) or (i >= 12 and i < 18) or (i >= 24 and i < 30) or (i >= 36 and i < 42) or (i >= 48 and i < 54) or (i >=60  and i < 66) or (i >= 72 and i < 78) or (i >= 84 and i < 90) or (i >= 96 and i < 102)):
		if (col == 0):
			worksheetP ['D' + str(4+row)] = dataA[i]
		elif (col == 1):
			worksheetP ['E' + str(4+row)] = dataA[i]
		elif (col == 2):
			worksheetP ['F' + str(4+row)] = dataA[i]	
		elif (col == 3):
			worksheetP ['H' + str(4+row)] = dataA[i]
		elif (col == 4):
			worksheetP ['I' + str(4+row)] = dataA[i]	
		elif (col == 5):
			worksheetP ['J' + str(4+row)] = dataA[i]	
		elif (col == 6):
			worksheetP ['L' + str(4+row)] = dataA[i]	
		elif (col == 7):
			worksheetP ['M' + str(4+row)] = dataA[i]
		elif (col == 8):
			worksheetP ['N' + str(4+row)] = dataA[i]			
	else:
		if (col == 0):
			worksheetP ['D' + str(6+row)] = dataA[i]
		elif (col == 1):
			worksheetP ['E' + str(6+row)] = dataA[i]
		elif (col == 2):
			worksheetP ['F' + str(6+row)] = dataA[i]	
		elif (col == 3):
			worksheetP ['H' + str(6+row)] = dataA[i]
		elif (col == 4):
			worksheetP ['I' + str(6+row)] = dataA[i]	
		elif (col == 5):
			worksheetP ['J' + str(6+row)] = dataA[i]	
		elif (col == 6):
			worksheetP ['L' + str(6+row)] = dataA[i]	
		elif (col == 7):
			worksheetP ['M' + str(6+row)] = dataA[i]
		elif (col == 8):
			worksheetP ['N' + str(6+row)] = dataA[i]	

myworkbook.save(filename)