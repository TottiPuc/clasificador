import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pickle

df=pd.read_csv('storepurchasedata.csv')
print(df.describe())

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=0)

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric= 'minkowski', p=2)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,-1]

print(y_test, y_pred, y_prob)

cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))

new_predict = classifier.predict(sc.transform(np.array([[40,20000]])))
print(new_predict)

new_predict = classifier.predict(sc.transform(np.array([[40,100000]])))
print(new_predict)

#picking the model and standar scaler

model_file = "clasificador.pickle"
pickle.dump(classifier, open(model_file,'wb'))
scalar_file = "sc.pickle"
pickle.dump(sc,open(scalar_file,'wb'))