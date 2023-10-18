import pickle
from flask import Flask, jsonify, request
from distutils.log import debug
from flask import Flask
from waitress import serve

input_file = f'model_C=1.0.bin'
with open(input_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask(__name__)
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
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# our query from marketing service will be in json and our response will also will be in json
# so we convert json into dictionary

@app.route('/predictX', methods=['POST'])
def predict_churn():
    customer = request.get_json()
    X = dv.transform(customer)
    Y_pred = model.predict_proba(X)[:, 1]
    res = Y_pred >= 0.5

    return jsonify({'Prob that customer churns': float(Y_pred),
                    'Churn': bool(res)})
# convert whole data in python form bcause json wont be able to recognize th numpy variartions.
# suppose we have Y_pred and res here.They are numpy float and boolean res. but we need to type case them
# into respective float() and bool() variables of python


# when we deploy code for production (in gunicorn/waitress) the below line will not be executed because paramteres are given ac to that
# while production
# gunicorn --bind 127.0.0.1/port_no filename:app
if __name__ == '__main__':
    serve(app, host="127.0.0.1", port=5000)  #for waitress
    # app.run(debug=True)

#print(f'Customer {customer}')
#print(f'Prob that he will churn{Y_pred}')


# this function is hard to get called from browser becasue browser sends GET request but we have POST request here
