import json
import requests

#url="http://127.0.0.1:8000/model" # para local
#url="http://35.202.153.111:8000/model" # para googlecloud
url = "http://c970669fcf11.ngrok.io/predict"

request_data=json.dumps({'age':'40','salary':'2000'})
response = requests.post(url,request_data)
print(response.text)