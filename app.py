from flask import Flask, request
import pickle
import numpy as np

local_classifier = pickle.load(open('clasificador.pickle','rb'))
local_scaler = pickle.load(open('sc.pickle','rb'))


app = Flask(__name__)

@app.route('/model',methods=['POST'])

def main():
    request_data=request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    prediction = local_classifier.predict(local_scaler.transform(np.array([[age, salary]])))

    return 'the prediction from GCP API is is {}'.format(prediction)

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8000, debug=True)
    #app.run(port=8000, debug=True)