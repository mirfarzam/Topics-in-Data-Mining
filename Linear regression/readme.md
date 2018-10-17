#Linear Regression

It is one of the most widely known modeling technique. Linear regression is usually among the first few topics which people pick
while learning predictive modeling. In this technique, the dependent variable is continuous, independent variable(s)
can be continuous or discrete, and nature of regression line is linear.

Linear Regression establishes a relationship between dependent variable (Y) and one or more independent variables (X)
using a best fit straight line (also known as regression line).

#Ridge Regression

Ridge Regression is a technique used when the data suffers from multicollinearity ( independent variables are highly correlated).
In multicollinearity, even though the least squares estimates (OLS) are unbiased, their variances are large which deviates the observed value
far from the true value. By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors.

Above, we saw the equation for linear regression. Remember? It can be represented as:

y=a+ b*x

This equation also has an error term. The complete equation becomes:
y=a+b*x+e (error term), [error term is the value needed to correct for a prediction error between the observed and predicted value]
=> y=a+y= a+ b1x1+ b2x2+…+e, for multiple independent variables.

In a linear equation, prediction errors can be decomposed into two sub components. First is due to the biased and second is due to the variance. Prediction error can occur due to any one of these two or both components. Here, we’ll discuss about the error caused due to variance.

Ridge regression solves the multicollinearity problem through shrinkage parameter λ (lambda). Look at the equation below.

###Ridge

In this equation, we have two components. First one is least square term and other one is lambda of the summation of β2 (beta- square) where β is the coefficient. This is added to least square term in order to shrink the parameter to have a very low variance.

Important Points:
•The assumptions of this regression is same as least squared regression except normality is not to be assumed
•It shrinks the value of coefficients but doesn’t reaches zero, which suggests no feature selection feature
•This is a regularization method and uses l2 regularization.

#Lasso Regression

Similar to Ridge Regression, Lasso (Least Absolute Shrinkage and Selection Operator) also penalizes the absolute size of the regression coefficients.
In addition, it is capable of reducing the variability and improving the accuracy of linear regression models.
Look at the equation below: LassoLasso regression differs from ridge regression in a way that it uses absolute values in the penalty function,
instead of squares. This leads to penalizing (or equivalently constraining the sum of the absolute values of the estimates)
values which causes some of the parameter estimates to turn out exactly zero. Larger the penalty applied,
further the estimates get shrunk towards absolute zero. This results to variable selection out of given n variables.

Important Points:
•The assumptions of this regression is same as least squared regression except normality is not to be assumed
•It shrinks coefficients to zero (exactly zero), which certainly helps in feature selection
•This is a regularization method and uses l1 regularization
•If group of predictors are highly correlated, lasso picks only one of them and shrinks the others to zero

---

#This project

### Linear regression on relationship of scores and sales

Firstly, I run the linear regression on relatioship of sales and scores. It has great amount of error. Because there is no logical relation between these two item. One game maybe has great score but it is not sold very much and exactly the reverse version of this senario is possible too. So I get about 70% error.

### Linear Regression on relationship of sales in North America & Europe and Global Sales 

It was a logical relationship. Amount of Sales in North America and Europe can guaruntee the global sales. In this calculation, the RMSE was 0.01466 and it is satisfiable.

### Ridge and Lasso on relationship on relationship of sales in North America & Europe and Global Sales

Based on Defenition of Ridge and Lasso and try both of them but it was obvious that Ridge was better method because i got lower RMSE in alfa = 1 in both algorithm. by alfa = 0.0002, RMSE in both is equal.


##Regression coefficients

Regression coefficients are estimates of the unknown population parameters and describe the relationship between a predictor variable and the response. In linear regression, coefficients are the values that multiply the predictor values.

The sign of each coefficient indicates the direction of the relationship between a predictor variable and the response variable.

- A positive sign indicates that as the predictor variable increases, the response variable also increases.
- A negative sign indicates that as the predictor variable increases, the response variable decreases.

The coefficient value represents the mean change in the response given a one unit change in the predictor.


#### in this project 

coef : [[1.14871181 1.32148989]]

and it means Europe Sales has more effect of total sales.

