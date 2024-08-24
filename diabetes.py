import pandas as pd
import numpy as np
df=pd.read_csv('diabetes_dataset.csv')

df.drop(columns=['year','gender','location','race:AfricanAmerican','race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other'], inplace=True)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['smoking_history'] = le.fit_transform(df['smoking_history'])

summary=df.describe()
# print(summary)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,confusion_matrix

x=df.drop(columns=['diabetes'])
y=df['diabetes']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

adb = AdaBoostClassifier()
adb_model = adb.fit(x_train,y_train)
y_adb_pred=adb_model.predict(x_test)
adb_model.score(x_train,y_train)
adb_accuracy = accuracy_score(y_test, y_adb_pred)
adb_f1 = f1_score(y_test, y_adb_pred)
adb_roc_auc = roc_auc_score(y_test, y_adb_pred)
adb_confusion = confusion_matrix(y_test, y_adb_pred)
print(f'accuracy: {adb_accuracy}')
print(f'f1 score: {adb_f1}')
print(f'roc_auc: {adb_roc_auc}')
print(f'confusuion matrix: \n{adb_confusion}')

import pickle
pickle.dump(adb_model,open('adb_model.pkl','wb'))

with open('label.pkl','wb') as f:
    pickle.dump(le,f)


pickle.dump(scaler,open('scalar.pkl','wb'))