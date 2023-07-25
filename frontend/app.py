from flask import Flask, render_template

app = Flask(__name__)


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def landing():
    return render_template('landing.html')


if __name__ == '__main__':
    app.run(port = '5000', debug=True)
