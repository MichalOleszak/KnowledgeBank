from flask import Flask

# Create an instance of the Flask class and pass in the "name" variable (which is filled by Python itself).
# This variable will be "main", if this file is being directly run through Python as a script. If you imported the file
# instead, the value of "name" would be the name of the file which you imported. For example, if you had test.py
# and run.py, and you imported test.py into run.py the "name" value of test.py will be test (app = Flask(test)).
app = Flask(__name__)

# route() is a decorator that tells Flask what URL should trigger the function defined as hello()
@app.route("/")
def hello():
    return "Hi!"


if __name__ == '__main__':
    app.run(debug=True)

# By default, flask runs on port 5000. If there is a port conflict, run:
# app.run(debug=True, port=12345)

