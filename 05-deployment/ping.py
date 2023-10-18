from flask import Flask, jsonify, request
from distutils.log import debug
from flask import Flask

# creating a Flask app
app = Flask(__name__)
# add decorator for extra functionality


# this app route takes this/service as path in url after localhost & exeutes immediate function after that
@app.route('/service', methods=['GET'])
def ping():
    return "web service accessed"

 # Using flask to make an api
# import necessary libraries and functions


# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.

@app.route('/', methods=['GET', 'POST'])
def home():
    if(request.method == 'GET'):

        data = "hello world"
        return jsonify({'data': data})


# A simple function to calculate the square of a number
# the number to be squared is sent in the URL when we use GET
# on the terminal type: curl http://127.0.0.1:5000 / home / 10
# this returns 100 (square of 10)
@app.route('/<int:num>', methods=['GET'])
def disp(num):

    return jsonify({'data': num**2})


# driver function
if __name__ == '__main__':

    app.run(debug=True)

# base image
# thisdir will be created
# we dont need virtual env here so we dont run pipenv here
# rather we install lib which re mentioned in pipfile & pipfile.lock
