# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

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
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: POOJA.S
RegisterNumber: 212223040146 
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
df=pd.read_csv('/content/Mammal_Cart.csv')
data=df.copy()
data.describe()
label_encoder=LabelEncoder()
data['Toothed']=label_encoder.fit_transform(data['Toothed'])
data['Hair']=label_encoder.fit_transform(data['Hair'])
data['Breathes']=label_encoder.fit_transform(data['Breathes'])
data['Legs']=label_encoder.fit_transform(data['Legs'])
data['Species']=label_encoder.fit_transform(data['Species'])
x=data.drop('Species',axis=1)
y=data['Species']
clf=DecisionTreeClassifier(criterion="entropy")
clf.fit(x,y)
plt.figure(figsize=(18,6))
plot_tree(clf,feature_names=x.columns,class_names=['Reptiles','Mammal'],filled=True)
plt.show()
```

## Output:
![Screenshot 2024-04-02 104707](https://github.com/poojasen05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150784373/520bbbc2-659f-41e3-9788-d86d438f0597)

![Screenshot 2024-04-02 104717](https://github.com/poojasen05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150784373/9998f937-4ded-4282-99b7-8e1ea76e61da)
![Screenshot 2024-04-02 104738](https://github.com/poojasen05/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150784373/6c458d45-f663-4b7b-9208-01cf9048e55f)





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

