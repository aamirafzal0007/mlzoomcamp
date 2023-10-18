

import requests


url = 'http://127.0.0.1:5000/credit_prob'
client = {'job': 'unknown', 'duration': 270, 'poutcome': 'failure'}
response=requests.post(url, json=client)
print(response)
responses = requests.post(url, json=client).json()
res = responses['Credit Probability']
print(res)
