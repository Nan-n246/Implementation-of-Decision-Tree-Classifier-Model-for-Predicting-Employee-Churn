# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step-1: Import the required libraries.

Step-2: Load the dataset.

Step-3: Define X and Y array.

Step-4: Define a function for costFunction,cost and gradient.

Step-5: Define a function to plot the decision boundary.

step 6: Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Nandhini.S
RegisterNumber: 212224230174
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
print(x.head())

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![image](https://github.com/user-attachments/assets/44c9bf71-c852-4dfd-91ca-21dc4a967b94)


![image](https://github.com/user-attachments/assets/0ad7c765-db41-496d-96e4-6a840809a053)

![image](https://github.com/user-attachments/assets/b7981d82-412f-44f2-bbeb-7fc9846ee118)

# Accuracy:

![image](https://github.com/user-attachments/assets/c2b06629-a81d-4c92-966a-2664190c26a0)

# Prediction:

![image](https://github.com/user-attachments/assets/da5d0f44-5063-44be-ab1f-0ec7467308c3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
