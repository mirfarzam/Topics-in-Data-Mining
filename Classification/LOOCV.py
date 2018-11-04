from sklearn.model_selection import KFold, LeaveOneOut
import numpy as np
from sklearn import metrics
import pandas as pd

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LOOCV = LeaveOneOut()

raw_data = pd.read_csv("./Biomechanical features of orthopedic patients.csv") 
raw_data  = raw_data.sample(frac=1).reset_index(drop=True)
inputs = raw_data[['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis']]
outputs = raw_data[['class']]
linear_Regression_Dictionary = {"Normal" : 1 , "Abnormal" : 0}
outputs = outputs.replace({"class": linear_Regression_Dictionary})

reverting = lambda x: 1 if(x>0.5) else 0

for train_index, test_index in LOOCV.split(inputs):
    X_train, X_test = inputs.iloc[train_index], inputs.iloc[test_index]
    y_train, y_test = outputs.iloc[train_index], outputs.iloc[test_index]


    lda_classification = LinearDiscriminantAnalysis()
    lda_classification.fit(X_train, y_train)
    lda_prediction = lda_classification.predict(X_test) 
    lda_finalPrediction = pd.DataFrame(np.array([reverting(xi) for xi in lda_prediction]))
    lda_MAE = metrics.mean_absolute_error(y_test, lda_finalPrediction)
    lda_MSE = metrics.mean_squared_error(y_test, lda_finalPrediction)
    lda_RMSE = np.sqrt(metrics.mean_squared_error(y_test, lda_finalPrediction))