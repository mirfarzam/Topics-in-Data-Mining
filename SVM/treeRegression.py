from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from utils import print_errors, R2
from dataset.dataset import get_regression_dataset

kf = KFold(n_splits=5)

raw_data = pd.read_csv("./Biomechanical features of orthopedic patients.csv") 
raw_data  = raw_data.sample(frac=1).reset_index(drop=True)
inputs = raw_data[['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis']]
outputs = raw_data[['class']]
linear_Regression_Dictionary = {"Normal" : 1 , "Abnormal" : 0}
outputs = outputs.replace({"class": linear_Regression_Dictionary})

reverting = lambda x: 1 if(x>0.5) else 0

for train_index, test_index in kf.split(inputs):
    X_train, X_test = inputs.iloc[train_index], inputs.iloc[test_index]
    y_train, y_test = outputs.iloc[train_index], outputs.iloc[test_index]


def treeRegression(tree_depth, x_train, x_test, y_train, y_test):
    regr = DecisionTreeRegressor(max_depth=tree_depth)
    regr.fit(x_train, y_train)
    return R2(regr.predict(x_test), y_test)


def engine():
    best_r2 = None
    best_depth = None

    for i in range(2, 20):
        r2 = main(i)
        if best_r2 is None or r2 < best_r2:
            best_r2 = r2
            best_depth = i
    x_train, x_test, y_train, y_test = get_regression_dataset()
    regr = DecisionTreeRegressor(max_depth=best_depth)
    regr.fit(x_train, y_train)
    print_errors(regr, x_train, y_train, x_test, y_test, msg='Tree-Based Regression')