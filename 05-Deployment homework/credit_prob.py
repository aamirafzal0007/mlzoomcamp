import pickle
from flask import Flask, jsonify, request
from distutils.log import debug
from waitress import serve


model_file = 'model1.bin'
dv_file = f'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

customer = {"job": "retired", "duration": 445, "poutcome": "success"}


app = Flask(__name__)


@app.route('/credit_prob', methods=['POST'])
def predict():
    customer = request.get_json()
    x = dv.transform(customer)
    y_pred = model.predict_proba(x)[:, 1]
    return jsonify({'Credit Probability': float(y_pred)})


if __name__ == '__main__':
    app.run(debug=True)
# print(y_pred)
