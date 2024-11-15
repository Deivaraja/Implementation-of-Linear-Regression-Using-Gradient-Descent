# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Deivaraja R
RegisterNumber: 24901238 
*/import numpy as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    x=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*x.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv",header=None)
print(data.head())
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
Scaler=StandardScaler()
y=(data.iloc[1:-1].values).reshape(-1,1)
X1_Scaled=Scaler.fit_transform(X1)
Y1_Scaled=Scaler.fit_transform(y)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,417484.1]).reshape(-1,1)
new_Scaled=Scaler.fit_transform(new_data)
prediction=np.dot (np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre:Scaler.inverse_transform(prediction)
print(f"predicted value: (pre)")

```

## Output:
![linear regression using gradient descent](sam.png)
![WhatsApp Image 2024-11-15 at 08 16 09_419d8112](https://github.com/user-attachments/assets/2116ab48-b8ed-4866-82e9-0eb15c47ace4)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
