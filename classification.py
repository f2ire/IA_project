##############################################
import pandas as pd

#data = pd.read_csv("data.txt", sep="\t")
#valeurmanquante=data.isna()
#valeurmanquante.sum(axis=0)
#print(valeurmanquante)
##############################################
#import
import pandas as pds
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
#label by discretisation of an attribute
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
data = pd.read_csv("data.txt", sep="\t")
data = data.drop(data.index[-5:])
print(data.tail())
# on a un tableau de données nommé "data" et je discrétiser la colonne d'indice 0

# supposons que vous avez un tableau de données nommé "X"
# et que vous souhaitez discrétiser la colonne d'indice 0

#centroids=km.cluster_centers_ # get the cluster centers
#print(centroids)
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal',strategy='uniform')
#'ordinal': les données sont encodées comme des entiers, où chaque bin est assigné à un entier unique.
#'onehot': les données sont encodées sous forme de vecteur de taille n_bins, avec une valeur de 1 dans le bin correspondant et 0 dans les autres.
#'binary': les données sont encodées comme des nombres binaires, avec une valeur de 1 dans le bin correspondant et -1 dans les autres.
print(discretizer)
#avec un data numerique
data= [[-2, 1, -4,   -1], [-1, 2, -3, -0.5],[ 0, 3, -2,  0.5],[ 1, 4, -1,    2]]
a=discretizer.fit(data)# compute the clusters
print(a)
objects = discretizer.transform(data)
print(objects)
#objects = discretizer.fit_transform(data[:,0].reshape(-1, 1))

km=KMeans(n_clusters=3) # create a KMeans object

labels = km.predict()
#labels = kmeans.fit_predict(X_binned)

from sklearn.linear_model import LogisticRegression

# entraînez un modèle de régression logistique en utilisant les étiquettes comme étiquettes de classe
clf = LogisticRegression()
clf.fit(data, labels)
