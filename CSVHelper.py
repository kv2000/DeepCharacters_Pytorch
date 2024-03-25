"""
@File: CSVHelper.py
@Date: Long long ago.
@Desc: CSVHelper.py by Dr. Marc Habermann; super cool stuff so that I decided directly to copy it from the original DeepCharacters Repo.
"""


########################################################################################################################
# Imports
########################################################################################################################

import numpy as np

########################################################################################################################
# 2D csv
########################################################################################################################
# Okey there is some minor changes
def load_csv_sequence_2D(filePath, type='int', skipRows=0, skipColumns=0, end_frame = 1145141919):
    
    csvfile =  open(filePath, 'rb')
    dataset = []
    length = 0
    lineCounter = 0
    realCounter = 0
    for line in csvfile:
        if lineCounter >= skipRows:
            values = line.split()
            length = len(values) - skipColumns
            for column, valueString in enumerate(values):
                if column >=skipColumns:
                    if(type == 'int'):
                        value = int(valueString)
                    else:
                        value = float(valueString)
                    dataset.append(value)
        lineCounter = lineCounter + 1
        realCounter += 1
        if realCounter > (end_frame):
            break
    dataset = np.array(dataset)
    dataset=np.reshape(dataset,(-1,length))
    csvfile.close()
    return dataset

########################################################################################################################
# 3D csv
########################################################################################################################

def load_csv_sequence_3D(filePath, batch1, type='int'):

    csvfile = []

    for b in range(0,batch1):
        csvfile.append(open(filePath + str(b) + '.csv', 'rb'))

    dataset = []
    lineLength = 0

    #go over the lines in all files in parallel
    while 1:

        #go over the files
        for b in range(0, batch1):

            line = csvfile[b].readline()
            if not line:
                break

            values = line.split()
            lineLength = len(values)

            for valueString in values:

                if (type == 'int'):
                    value = int(valueString)
                else:
                    value = float(valueString)

                dataset.append(value)
        if not line:
            break

    dataset = np.array(dataset)
    dataset = np.reshape(dataset, (-1, batch1, lineLength))

    return dataset


########################################################################################################################
# 4D csv
########################################################################################################################

def load_csv_sequence_4D(filePath, batch1, batch2, batch3, type='int'):

    csvfile = []

    for b in range(0, batch1):
        csvfile.append(open(filePath + str(b) + '.csv', 'rb'))

    dataset = []

    # go over the lines in all files in parallel
    counter =0
    while 1:

        # go over the files
        for b in range(0, batch1):

            line = csvfile[b].readline()

            if not line:
                break

            values = line.split()

            for valueString in values:

                if (type == 'int'):
                    try:
                        value = int(valueString)
                    except ValueError:
                        print(line)
                else:
                    value = float(valueString)

                dataset.append(value)

        if not line:
            break
        counter = counter + 1

    dataset = np.array(dataset)
    dataset = np.reshape(dataset, (-1, batch1, batch2, batch3))

    return dataset

########################################################################################################################
# 4D csv compact
########################################################################################################################

def load_csv_compact_4D(filePath, batch1, batch2, batch3, skipRows=0, skipColumns=0, type='int', end_frame = 1145141919):

    csvfile = open(filePath, 'rb')
    dataset = []
    lineCounter = 0
    realCounter = 0

    # go over the lines in the file
    while 1:

        line = csvfile.readline()

        if not line:
            break

        #skip first rows
        if lineCounter >= skipRows:

            values = line.split()
            for c2, valueString in enumerate(values):
                
                #skip columns
                if c2 >= skipColumns:

                    if (type == 'int'):
                        value = int(valueString)
                    else:
                        value = float(valueString)

                    dataset.append(value)
                    

        lineCounter = lineCounter + 1   
        realCounter += 1
        if realCounter > (end_frame):
            break

    dataset = np.array(dataset)
    dataset = np.reshape(dataset, (-1, batch1, batch2, batch3))
    csvfile.close()
    return dataset

