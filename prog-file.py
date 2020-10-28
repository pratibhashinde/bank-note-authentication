import pandas as pd
import numpy as np


df=pd.read_csv('data.csv')
#print(df.head())

#input and output data
x=df.drop('class',axis=1)
y=df['class']

#split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#random forest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#evaluation
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

print(model.predict([[2,3,4,1]]))

#saving model in pickle format
import pickle
a=open('model.pkl','wb')
pickle.dump(model,a)
a.close()




