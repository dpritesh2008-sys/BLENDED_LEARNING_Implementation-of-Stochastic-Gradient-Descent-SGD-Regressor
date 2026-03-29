# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize Model – Set initial values for weights and bias.

2.Pick One Sample – Select one data point randomly from the dataset.

3.Calculate Error – Find the difference between actual and predicted value.

4.Update Weights – Adjust weights using learning rate and gradient.

5.Repeat Process – Continue for many iterations until the model improves.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: Ritesh DP
RegisterNumber:  212225040339
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

#Load the dataset
data=pd.read_csv("CarPrice_Assignment (4).csv")
print(data.head())
print(data.info())

#Data Processing
#Dropping unnecessary columns and handling categorical variables
data=data.drop(['CarName','car_ID'], axis=1)
data=pd.get_dummies(data,drop_first=True)

#Splitting the data into features and target variables
x=data.drop('price',axis=1)
y=data['price']

#Standardizing the data
scaler = StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

#Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Creating the SGD Regressor model
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(x_train,y_train)

#Making predictions
y_pred=sgd_model.predict(x_test)

#Evaluating model performance
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

#Print evaluation metrics
print('Name: Balasurya S')
print('Reg. No: 25000944')
print("Mean Squared Error:",mse)
print("R-squared Score:",r2)
print("Mean Absolute Error:",mae)

#Print model coefficients
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

#Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red') #Perfect prediction line
plt.show()


```


## Output:
<img width="593" height="383" alt="image" src="https://github.com/user-attachments/assets/72bc4cf7-f52b-42ce-b088-b09a0e1a5af6" />
<img width="491" height="510" alt="image" src="https://github.com/user-attachments/assets/5b86d8f4-48cd-461b-b10e-a196585ee6f1" />
<img width="845" height="123" alt="image" src="https://github.com/user-attachments/assets/9b228b9a-6b3a-433d-a249-e74688a8018d" />
<img width="341" height="81" alt="image" src="https://github.com/user-attachments/assets/061e857a-3cba-4550-94bd-1e31eca942e1" />
<img width="682" height="177" alt="image" src="https://github.com/user-attachments/assets/9beeaa1e-3215-47ec-bce8-9efc0513a050" />
<img width="565" height="453" alt="download" src="https://github.com/user-attachments/assets/dc5acb16-3683-41e5-ab04-ac47ec60594e" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
