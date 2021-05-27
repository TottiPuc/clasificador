import pickle
import numpy as np

local_classifier = pickle.load(open('clasificador.pickle','rb'))
local_scaler = pickle.load(open('sc.pickle','rb'))

new_pred= local_classifier.predict(local_scaler.transform(np.array([[22,122000]])))
print(new_pred)
new_prob= local_classifier.predict_proba(local_scaler.transform(np.array([[22,122000]])))[:,-1]
print(new_prob)