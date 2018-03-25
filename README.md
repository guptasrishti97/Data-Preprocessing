# Data-Preprocessing
#In machine learning we need to import our dataset in such a way that further processing can be done on the required data
#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
df=pd.DataFrame(X)
Y=dataset.iloc[:,3].values
df1=pd.DataFrame(Y)

#Missing Data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y) 

#Splitting Dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

