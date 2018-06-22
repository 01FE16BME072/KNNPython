import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

def KNearestNeighbors(data,predict,k = 3):
	if len(data) >= k:
		warnings.warn('Hey you have choosen k value less then the length of the data dumbass')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean,group])

	votes = [i[1] for i in sorted(distances)[:k]]
	global votes_count
	votes_count = Counter(votes).most_common(1)[0][0]
	#print(votes_count)
	return votes_count

dataframe = pd.read_csv('cancer.csv')

dataframe.replace('?',-99999,inplace = True)
dataframe.drop(['id'],1,inplace = True)

#print(dataframe.head())
#print(full_data[:10])

full_data = dataframe.astype(float).values.tolist()

random.shuffle(full_data)
# 2 is banign 4 is Malignant
test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])


correct = 0.0
total = 0.0

for group in test_set:
	for data in test_set[group]:
		vote = KNearestNeighbors(train_set,data,k = 5)
		if group == vote:
			correct +=1
		total+=1

#print(correct)
#print(total)

Accuracy = correct/total


print 'Accuracy: ', Accuracy

pred = KNearestNeighbors(test_set,[5,1,1,1,2,1,3,1,1],k = 3)
print(votes_count)

