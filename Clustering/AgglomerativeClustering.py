import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
import numpy as np

raw_data = pd.read_csv("./Pokemon.csv") 
raw_data  = raw_data.sample(frac=1).reset_index(drop=True)
inputs = raw_data[['HP','Attack','Defense','Speed']]
inputs_train = inputs.loc[:600]
inputs_test = inputs.loc[600:]


#“ward”, “complete”, “average”, “single”

clusteringWard = AgglomerativeClustering(linkage='ward').fit(inputs_train) 
clusteringComplete = AgglomerativeClustering(linkage='complete').fit(inputs_train) 
clusteringAverage = AgglomerativeClustering(linkage='average').fit(inputs_train) 
clusteringSingle = AgglomerativeClustering(linkage='single').fit(inputs_train) 
