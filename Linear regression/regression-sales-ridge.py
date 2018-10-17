
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
import seaborn as sns



raw_data = pd.read_csv("./video-games.csv") 


filtered_data_Stage1 = raw_data[~np.isnan(raw_data['User_Score'])]
filtered_data_Stage2 = filtered_data_Stage1[~np.isnan(filtered_data_Stage1['Critic_Score'])]
data = filtered_data_Stage2[~np.isnan(filtered_data_Stage2['Global_Sales'])]
df1 = data[['NA_Sales','EU_Sales','Global_Sales']]

inputs = df1[['NA_Sales','EU_Sales']]
outputs = df1[['Global_Sales']]

inputs_train = inputs.loc[:15000]
inputs_test = inputs.loc[15000:]

outputs_train = outputs.loc[:15000]
outputs_test = outputs.loc[15000:]

lin_regr_ridge = linear_model.Ridge(alpha=0.5)
lin_regr_lasso = linear_model.Lasso(alpha=0.0002)

lin_regr_ridge.fit(inputs_train, outputs_train)
lin_regr_lasso.fit(inputs_train, outputs_train)


prediction_ridge = lin_regr_ridge.predict(inputs_test)

prediction_lasso = lin_regr_lasso.predict(inputs_test)


print('Ridge - MAE:', metrics.mean_absolute_error(outputs_test, prediction_ridge))
print('Lasso - MAE:', metrics.mean_absolute_error(outputs_test, prediction_lasso))
print('\n///////////\n')
print('Ridge - MSE:', metrics.mean_squared_error(outputs_test, prediction_ridge))
print('Lasso - MSE:', metrics.mean_squared_error(outputs_test, prediction_lasso))
print('\n///////////\n')
print('Ridge - RMSE:', np.sqrt(metrics.mean_squared_error(outputs_test, prediction_ridge)))
print('Lasso - RMSE:', np.sqrt(metrics.mean_squared_error(outputs_test, prediction_lasso)))



# plt.scatter(inputs_test[['Critic_Score']], outputs_test,  color='black')
# plt.title('Test Data')
# plt.xlabel('Size')
# plt.ylabel('Price')
# plt.xticks(())
# plt.yticks(())
# plt.plot(inputs_test[['Critic_Score']], prediction, color='red',linewidth=3)
# plt.show()



