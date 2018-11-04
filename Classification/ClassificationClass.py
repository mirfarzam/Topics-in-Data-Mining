import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.model_selection import KFold

class LDA():

    def __init__(self, inputs, outputs,model):
        self.inputs = inputs
        self.outputs = outputs
        self.model = model

    def split(self,index):
        inputs_train = self.inputs.loc[:index]
        inputs_test = self.inputs.loc[index:]
        outputs_train = self.outputs.loc[:index]
        outputs_test = self.outputs.loc[index:]

    def kfoldSplit(self,train_index,test_index):
            X_train, X_test = self.inputs.iloc[train_index], self.inputs.iloc[test_index]
            y_train, y_test = self.outputs.iloc[train_index], self.outputs.iloc[test_index]
            return X_train,X_test,y_train,y_test

    def kfoldEngine(self,k):
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(self.inputs):
            X_train, X_test , y_train, y_test = self.kfoldSplit(train_index,test_index)


    # def core(self):

    #     # switch(self.modelName):
        

    #     model = LinearDiscriminantAnalysis()
    #     lda_classification.fit(self.inputs_train, outputs_train)    
    
    def Engine(self):
        lda_classification = LinearDiscriminantAnalysis()
        lda_classification.fit(self.inputs_train, self.outputs_train)
        prediction = lda_classification.predict(self.inputs_test)
        reverting = lambda x: 1 if(x>0.5) else 0
        self.prediction = pd.DataFrame(np.array([reverting(xi) for xi in prediction]))

    def prediction(self,model,inputs_test,outputs_test):
        prediction = model.predict(inputs_test)
        reverting = lambda x: 1 if(x>0.5) else 0
        self.prediction = pd.DataFrame(np.array([reverting(xi) for xi in prediction]))

    def getErrors(self):
        self.MAE = metrics.mean_absolute_error(self.outputs_test, self.prediction)
        self.MSE = (self.outputs_test, self.prediction)
        self.RMSE = np.sqrt(metrics.mean_squared_error(self.outputs_test, self.prediction))
