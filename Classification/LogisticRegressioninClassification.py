import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
import seaborn as sns



raw_data = pd.read_csv("./Biomechanical features of orthopedic patients.csv") 

raw_data  = raw_data.sample(frac=1).reset_index(drop=True)

inputs = raw_data[['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis']]
outputs = raw_data[['class']]




linear_Regression_Dictionary = {"Normal" : 1 , "Abnormal" : 0}

Reverse_linear_Regression_Dictionary = {1 : "Normal"  ,  0 : "Abnormal"}



output_linr = outputs.replace({"class": linear_Regression_Dictionary})

inputs_train = inputs.loc[:248]
inputs_test = inputs.loc[248:]
outputs_train = output_linr.loc[:248]
outputs_test = output_linr.loc[248:]

log_regr = linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

log_regr.fit(inputs_train, outputs_train)

prediction = log_regr.predict(inputs_test)

reverting = lambda x: 1 if(x>0.5) else 0
finalPrediction = pd.DataFrame(np.array([reverting(xi) for xi in prediction]))

print('MAE:', metrics.mean_absolute_error(outputs_test, finalPrediction))
print('MSE:', metrics.mean_squared_error(outputs_test, finalPrediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(outputs_test, finalPrediction)))
