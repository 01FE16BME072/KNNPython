from matplotlib import pyplot as plt
import numpy as np
import warnings
from collections import Counter

datasets = {'black':[[1,2],[2,3],[3,1]],'red':[[6,5],[7,7],[8,6]]}
newFeature = [3,7]

for i in datasets:
	for ii in datasets[i]:
		plt.scatter(ii[0],ii[1],s = 50,color = i)
plt.scatter(newFeature[0],newFeature[1],s = 50,color = 'yellow')
plt.show()

def K_Nearest_Neighbor(data,predict,k = 3):
	if len(data) >= k:
		warnings.warn('Hey you have choosen k value less then the length of the data dumbass')
	
	distances = []
	for group in data:
		for features in data[group]:
			euclidean = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean,group])

	votes = [i[1] for i in sorted(distances)[:k]]
	#print(votes)
	#print(Counter(votes).most_common(1))

	votes_result = Counter(votes).most_common(1)[0][0]

	#print(votes_result)
	
	return votes_result

k = K_Nearest_Neighbor(datasets,newFeature,k = 3)

for i in datasets:
	for ii in datasets[i]:
		plt.scatter(ii[0],ii[1],s = 50,color = i)
plt.scatter(newFeature[0],newFeature[1],s = 50,color = k)
plt.show()





