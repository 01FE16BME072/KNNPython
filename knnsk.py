import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

dataframe = pd.read_csv('cancer.csv')
#print(dataframe.head())
dataframe.replace('?',-99999,inplace = True)
dataframe.drop(['id'],1,inplace = True)

X = dataframe.drop(['CLass'],1)
Y = dataframe['CLass'] 

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.25)

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5,p = 2)
classifier.fit(X_train,Y_train)

Accuracy = classifier.score(X_test,Y_test)

print(Accuracy)

PredictThis = np.array([4,1,1,2,3,5,6,7,4])
PredictThis = PredictThis.reshape(1,-1)

Predict = classifier.predict(PredictThis)

print(Predict)

# cm = confusion_matrix(Y_test, classifier.predict(X_test))
# print(cm)