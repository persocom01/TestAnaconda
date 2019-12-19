from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'


# Makes runs the server when this python file is called.
# app.run(host='127.0.0.1', port=5000, debug=False, options)
# host='0.0.0.0' makes the server available externally. This is not advisable
# because in debugging mode a user can execute python code on ther server. Some
# host numbers appear to be invalid for some reason, such as '128.0.0.1'.
if __name__ == '__main__':
    app.run(host='127.0.0.2', port=5050)
