import openpyxl
import os
import math

dirname = os.path.realpath('.')
filename = dirname + '\\Summary of Good Tests.xlsx'
data = dirname + '\\Models&Results\\resultsFinal.txt'
myworkbook = openpyxl.load_workbook(filename)

worksheetP = myworkbook['Final Exercise Position']

f=open(data)
lines=f.readlines()

dataA = []
for line in lines:
	newLine = line.split(':')
	newLine = newLine[1].split('\n')
	dataA.append(float((newLine[0])))

print (dataA)
print (len(dataA))

for i in range (0, len(dataA)):
	col = math.floor(i / 72)
	row = i%72
	print(i)
	if ((i < 18) or (i >= 72 and i < 90) or (i >= 144 and i < 162)):
		if (col ==0):
			worksheetP ['E' + str(4+row)] = dataA[i]
		elif (col == 1):
			worksheetP ['F' + str(4+row)] = dataA[i]
		else:
			worksheetP ['G' + str(4+row)] = dataA[i]

	elif ((i >= 18 and i < 36) or (i >= 90 and i < 108) or (i >= 162 and i < 180)) :
		if (col ==0):
			worksheetP ['E' + str(7+row)] = dataA[i]
		elif (col == 1):
			worksheetP ['F' + str(7+row)] = dataA[i]
		else:
			worksheetP ['G' + str(7+row)] = dataA[i]

	elif ((i >= 36 and i < 54) or (i >= 108 and i < 126) or (i >= 180 and i < 198)):
		if (col ==0):
			worksheetP ['E' + str(10+row)] = dataA[i]
		elif (col == 1):
			worksheetP ['F' + str(10+row)] = dataA[i]
		else:
			worksheetP ['G' + str(10+row)] = dataA[i]

	elif ((i >= 54 and i < 72) or (i >= 126 and i < 144) or (i >= 198 and i < 216)):
		if (col ==0):
			worksheetP ['E' + str(13+row)] = dataA[i]
		elif (col == 1):
			worksheetP ['F' + str(13+row)] = dataA[i]
		else:
			worksheetP ['G' + str(13+row)] = dataA[i]

myworkbook.save(filename)