

import requests

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 5,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


url = 'http://127.0.0.1:5000/predictX'

requests.post(url, json=customer)

response = requests.post(url, json=customer).json()
if(response['Churn']):
    discount = 1.0-response['Prob that customer churns']
    discount = round(discount, 2)
    print(f'Send email to the customer with {discount*100} % off')
else:
    print("Discount email not need to be sent")
