import matplotlib.pyplot as plt

import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd

import seaborn as sns



raw_data = pd.read_csv("./video-games.csv") 


filtered_data_Stage1 = raw_data[~np.isnan(raw_data['User_Score'])]
filtered_data_Stage2 = filtered_data_Stage1[~np.isnan(filtered_data_Stage1['Critic_Score'])]
data = filtered_data_Stage2[~np.isnan(filtered_data_Stage2['Global_Sales'])]
df1 = data[['User_Score','Critic_Score','Global_Sales']]

inputs = df1[['User_Score','Critic_Score']]
outputs = df1[['Global_Sales']]

inputs_train = inputs.loc[:15000]
inputs_test = inputs.loc[15000:]

print(inputs_train)

outputs_train = outputs.loc[:15000]
outputs_test = outputs.loc[15000:]

print(outputs_train)

lin_regr = linear_model.LinearRegression()

lin_regr.fit(inputs_train, outputs_train)

prediction = lin_regr.predict(inputs_test)

print('MAE:', metrics.mean_absolute_error(outputs_test, prediction))
print('MSE:', metrics.mean_squared_error(outputs_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(outputs_test, prediction)))




# plt.scatter(inputs_test[['Critic_Score']], outputs_test,  color='black')
# plt.title('Test Data')
# plt.xlabel('Size')
# plt.ylabel('Price')
# plt.xticks(())
# plt.yticks(())
# plt.plot(inputs_test[['Critic_Score']], prediction, color='red',linewidth=3)
# plt.show()



