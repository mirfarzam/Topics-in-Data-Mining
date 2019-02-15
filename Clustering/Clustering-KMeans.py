import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
raw_data = pd.read_csv("./Pokemon.csv") 
raw_data  = raw_data.sample(frac=1).reset_index(drop=True)
inputs = raw_data[['HP','Attack','Defense','Speed']]
inputs_train = inputs.loc[:600]
inputs_test = inputs.loc[600:]
# Different variance
X_varied, y_varied = make_blobs(n_samples=600,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=150)
y_pred =  KMeans(5,init='random').fit_predict(inputs_test)