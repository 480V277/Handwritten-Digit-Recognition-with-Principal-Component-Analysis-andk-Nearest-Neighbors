def myCrossvalind(K,n):
    #randomly assign each of n examples to K groups such that each group has
    #approximately the same number of examples
   
    #example function call:
    #           groupingVector = myCrossvalind(5,200)
    # creates a vector of length 200 of 5 different numbers 0 through 4 such
    # that each number occurs exactly 40 times, but the occurrences are random.
   
    #copyright: Professor Paul M. Kump
    #last updated: 6/3/18
   
    import numpy as np
    import random
    import math
   
    remainder = n%K #in case n is not divisible by K
    integer = math.floor(n/K) #minimum number in each group
   
    indices = []
    for ii in np.arange(0,K): #integer part
        indices = indices + [ii]*integer
   
    for ii in np.arange(0,remainder): #remaining
        indices = indices + [ii]
   
    random.shuffle(indices)  #randomize
    indices = np.asarray(indices) #convert list to array
    return indices


def myCSVreader(filename):
    #reads in the csv file specified by FILENAME and returns the data in a
    #python list with the same dimensions as the csv file.  Good for non-
    #numerical data (i.e., not an np array).
   
    #Example call
        # dataset = myCSVreader('mushroom.csv')
   
    #copyright: Professor Paul M. Kump
    #last updated: 6/11/18        
   
    import csv

    rows = []   #initialize output
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
   
        for row in csvreader:
            rows.append(row)

    return rows


def myScatter3D(threeDdata, groupInfo):
    #makes a 3-D scatter plot of classified data
    #assumes groups are labeled as 0,1,2,3.  Up to 4 groups
   
    #example function call:
    #     myScatter3D(X, groupingVector)
    # makes a scatter plot of the data in X with the grouping information in
    # groupingVector.  X should have no fewer than three columns.  If more,
    # the function uses the first three columns by default.  groupingVector
    # should have no more than four different numbers.
   
    #copyright: Professor Paul M. Kump
    #last updated: 6/4/18
   
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import style
    style.use('ggplot')
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
   
    numGroups = len(set(groupInfo)) #set() gets unique elements
    colorSpec = ['b','r', 'g', 'k']
    markerSpec = ['^','.','s','D' ]
   
    for ii in range(numGroups):
        x = threeDdata[groupInfo == ii,0]
        y = threeDdata[groupInfo == ii,1]
        z = threeDdata[groupInfo == ii,2]
       
        ax.scatter(x,y,z,c=colorSpec[ii], marker = markerSpec[ii])
       
    ax.set_xlabel('First Dimension', fontsize = 18)    
    ax.set_ylabel('Second Dimension', fontsize = 18)
    ax.set_zlabel('Third Dimension', fontsize = 18)
   
    fig.suptitle('3-D Plot Showing Classes (Clusters)', fontsize = 26)
   
    plt.show()