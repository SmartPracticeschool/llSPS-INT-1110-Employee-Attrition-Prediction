# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('Employee-Attrition.csv')

x=df.iloc[:,1:35].values
y=np.array(df.iloc[:,0:1].values)
#y.reshape(1,-1)
y = y.ravel()

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
oh=OneHotEncoder(categories='auto')
x=oh.fit_transform(x).toarray() 
lb=LabelEncoder()
y=lb.fit_transform(y) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=45,criterion="entropy")
rf.fit(x_train,y_train)

pickle.dump(rf, open('model.pkl','wb'))
pickle.dump(oh,open('oneencoder.pkl','wb'))
pickle.dump(lb,open('labelencoder.pkl','wb'))