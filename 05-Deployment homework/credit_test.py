

import requests

host='ip172-18-0-13-cknvs5efml8g00bp8rt0-8000.direct.labs.play-with-docker.com'
url_docker=f'http://{host}/credit_prob'
url = 'http://127.0.0.1:5000/credit_prob'
client = {"job": "retired", "duration": 445, "poutcome": "success"}
response=requests.post(url_docker, json=client)
print(response)
responses = requests.post(url_docker, json=client).json()
res = responses['Credit Probability']
print(res)
