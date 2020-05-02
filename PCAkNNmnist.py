from ClassHelper import myCSVreader
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#get data
data_downloaded = myCSVreader('mnist_test.csv')
data_downloaded = np.array(data_downloaded)
data_targets = np.array([int(num_str) for num_str in \
                         data_downloaded[1:,0]])
data = np.array([[int(num_str) for num_str in inner] \
                  for inner in data_downloaded[1:,1:]] )


#each feature

pca = PCA(n_components = 784)
fit = pca.fit(data,data_targets)

var = fit.explained_variance_ratio_
var1 = np.cumsum(np.round(var, decimals=4)*100)
T = fit
W = fit.components_


pca_new = PCA(n_components = 36 )
T_new = pca_new.fit_transform(data)
pca_two = PCA(n_components = 2 )
T_two = pca_two.fit_transform(data)


zero = np.where(data_targets == 0)
one = np.where(data_targets  == 1)
two = np.where(data_targets  == 2)
three = np.where(data_targets  == 3)
four = np.where(data_targets  == 4)
five = np.where(data_targets  == 5)
six = np.where(data_targets  == 6)
seven = np.where(data_targets  == 7)
eight = np.where(data_targets  == 8)
nine = np.where(data_targets  == 9)
#---------------------------
t1 = T_two[:,0]
t2 = T_two[:,1]  

#class zero
D1_0 = t1[zero]
D2_0 = t2[zero]
#class one
D1_1 = t1[one]
D2_1 = t2[one]
#class two
D1_2 = t1[two]
D2_2 = t2[two]
#class three
D1_3 = t1[three]
D2_3 = t2[three]
#class four
D1_4 = t1[four]
D2_4 = t2[four]
#class five
D1_5 = t1[five]
D2_5 = t2[five]
#class six
D1_6 = t1[six]
D2_6 = t2[six]
#class seven
D1_7 = t1[seven]
D2_7 = t2[seven]
#class eight
D1_8= t1[eight]
D2_8 = t2[eight]
#class nine
D1_9= t1[nine]
D2_9 = t2[nine]

blue = plt.scatter(D1_0,D2_0,color='blue')
green = plt.scatter(D1_1,D2_1,color='green')
red = plt.scatter(D1_2,D2_2,color='red')
cyan = plt.scatter(D1_3,D2_3,color='cyan')
magenta = plt.scatter(D1_4,D2_4,color='magenta')
yellow = plt.scatter(D1_5,D2_5,color='yellow')
black = plt.scatter(D1_6,D2_6,color='black')
slategray = plt.scatter(D1_7,D2_7,color='slategray')
mediumspringgreen = plt.scatter(D1_8,D2_8\
,color='mediumspringgreen')
purple = plt.scatter(D1_9,D2_9,color='purple')

plt.title('Projected 2D Data',fontsize  = 25)
plt.tick_params(labelsize=12)
plt.xlabel('First Principal Component',fontsize = 20)
plt.ylabel('Second Principal Component',fontsize = 20)
plt.legend((blue,green,red,cyan,magenta,yellow,black\
,slategray,mediumspringgreen,purple),('Ones','Twos'\
,'Threes','Fours','Fives','Sixes','Sevens','Eights'\
,'Nines'), loc=0,fontsize = 12)
plt.show()





#-------Plotting the scree of variance in the pc
plt.figure()
plt.plot(var1)

plt.title('Scree Plot of Principal Components', fontsize  = 25)
plt.tick_params(labelsize=12)
plt.xlabel('Number of Principal Components',fontsize  = 20)
plt.ylabel('Percent Variance Retained', fontsize  = 20)
plt.show()
print('---------------------------')

#-------Images of regular and transformed
queryIm = T_new[3,:]  
queryIm.shape=(6,6)
plt.figure()
plt.imshow(255-queryIm,cmap='gray')
plt.title('Transformed 0 (36 Dimesnions)',fontsize  = 25)
plt.show()

queryIm = data[3,:]  
queryIm.shape=(28,28)
plt.figure()
plt.imshow(255-queryIm,cmap='gray')
plt.title('Regular 0 (784 Dimesnions)',fontsize  = 25)
plt.show()
print('---------------------------')
#-------------------------------------------------
queryIm = T_new[1,:]  
queryIm.shape=(6,6)
plt.figure()
plt.imshow(255-queryIm,cmap='gray')
plt.title('Transformed 2 (36 Dimesnions)',fontsize  = 35)
plt.show()

queryIm = data[1,:]  
queryIm.shape=(28,28)
plt.figure()
plt.imshow(255-queryIm,cmap='gray')
plt.title('Regular 2 (784 Dimensions)',fontsize  = 35)
plt.show()
print('---------------------------')
#------------36 Dimensions---------------------------
accur = []
neighbors = []
for i in range(1,10):
    knn =  KNeighborsClassifier(n_neighbors = i)
    prediction = cross_val_predict(knn,T_new,data_targets,cv = 5)
    C = confusion_matrix(data_targets, prediction)
    accur = np.append(accur,np.trace(C)/np.sum(C))
    neighbors = np.append(neighbors,i)
maximum = np.where(accur == np.max(accur))
maximum = np.array(maximum)
maximum_neighbors = neighbors[maximum]
print('---------------------------')
print('The number of neighbors that gives the highest \
      accuracy (36 Dimensions):',int(maximum_neighbors))
print('Accuracy(36 Dimensions):' \
      ,accur[int(maximum_neighbors -1)]*100,'%')
#-----------Two Dimensions----------------------------------
Accur = []
Neighbors = []
for j in range(1,10):
    KNN =  KNeighborsClassifier(n_neighbors = j)
    Prediction = cross_val_predict(KNN,T_two,data_targets ,cv = 5)
    c = confusion_matrix(data_targets, Prediction)
    Accur = np.append(Accur,np.trace(c)/np.sum(c))
    Neighbors = np.append(Neighbors,j)
Maximum = np.where(Accur == np.max(Accur))
Maximum = np.array(Maximum)
Maximum_neighbors = Neighbors[Maximum]
print('---------------------------')
print('The number of neighbors that gives the highest accuracy \
      (2 Dimensions):',int(Maximum_neighbors))
print('Accuracy (2 Dimensions):' \
      ,Accur[int(Maximum_neighbors -1)]*100,'%')
