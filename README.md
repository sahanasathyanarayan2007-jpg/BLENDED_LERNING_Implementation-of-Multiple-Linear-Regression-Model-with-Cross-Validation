# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Start

Import required libraries

Load the car price dataset

Preprocess the data

Split the dataset into training and testing sets

Create the Multiple Linear Regression model

Apply K-Fold Cross-Validation

Train the model

Predict car prices

Evaluate the model performance

Stop

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Sahana.s
RegisterNumber:  25004522
*/
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df=pd.read_csv('CarPrice_Assignment.csv')
#1. LOAD AND PREPARE DATA
data= df.drop(['car_ID', 'CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
#2.SPLIT DATA
X=data.drop('price', axis=1)
y=data['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
#3.CREATE AND TRAIN MODEL
model= LinearRegression()
model.fit(X_train,y_train)
print("Name:Sahana")
print("Reg. No:25004522")

print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(model, X, y, cv=5)

print("Fold R2 scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")

y_pred = model.predict(X_test)

print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE:{mean_absolute_error(y_test,y_pred):.2f}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```

## Output:
<img width="738" height="258" alt="image" src="https://github.com/user-attachments/assets/68cb623f-f1ea-4d8e-8564-da4af1b058a5" />
<img width="1047" height="697" alt="image" src="https://github.com/user-attachments/assets/4de62f0f-7f74-4965-b6b4-517fc574ec56" />




## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
