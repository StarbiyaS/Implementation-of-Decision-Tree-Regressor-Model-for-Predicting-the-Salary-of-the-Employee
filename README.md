# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Starbiya S
RegisterNumber: 212223040208 
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![Screenshot 2024-09-26 213834](https://github.com/user-attachments/assets/273f3672-8b91-4844-8400-8a54c1c8234b)


![Screenshot 2024-09-26 213843](https://github.com/user-attachments/assets/10584a1f-b6a5-4c58-926b-8d9fa83cc000)


![Screenshot 2024-09-26 213849](https://github.com/user-attachments/assets/a24a9bd8-6ada-4959-8754-03318c960c4f)


![Screenshot 2024-09-26 213856](https://github.com/user-attachments/assets/2ae78435-d56d-42a1-aa4d-c263788bca4b)



![Screenshot 2024-09-26 213916](https://github.com/user-attachments/assets/9c8867d7-d623-4234-a278-2531fb796124)




![Screenshot 2024-09-26 213926](https://github.com/user-attachments/assets/497d8e6a-2433-4b35-b523-7c3db0a47ee8)



![Screenshot 2024-09-26 213949](https://github.com/user-attachments/assets/9f4e8d6f-c23d-4c85-9223-15aacb42cd74)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
