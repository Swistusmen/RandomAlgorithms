import matplotlib.pyplot as plt
from sklearn  import cluster
import pandas as pd
import numpy as np

#example dataset
dataset= pd.DataFrame({"x":[11,21,28,17,29,33,24,45,45,52,51,52,55,53,55,61,62,70,72,10],'y':[39,36,30,52,53,46,55,59,63,70,66,63,58,23,14,8,18,7,24,10]})

myKmeans=cluster.KMeans(n_clusters=2)
myKmeans.fit(dataset)

#print("Środki klastrów: "+myKmeans.cluster_centers_)
#print("Elementy przypisane do klastra: "+myKmeans.labels_)
centroids=myKmeans.cluster_centers_
#drawing

plt.scatter(dataset['x'],dataset['y'],s=100)
plt.scatter(centroids[0],centroids[1],s=100)
plt.show()

