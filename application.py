from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from model_files.ml_model import predict_model
application = app = Flask(__name__)
CORS(application)


@app.route('/api/predict', methods=['POST'])
def predict():
    credit = request.get_json()
    print(credit)

    with open('./model_files/pkl/PYMEmodel.sav', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_model(credit, model)

    result = {
        'predictions': list(predictions)
    }

    return jsonify(result)
    # response = jsonify(credit)
    # return response


@app.route('/', methods=['GET'])
def welcome():
    result = jsonify({
        'team': 'inspirational',
        'members': ['Yennifer', 'Mary', 'Isidro', 'Edgar', 'Benjamin', 'Ernesto'],
        'solution': 'ML platform'
    })

    result.headers.add('Access-Control-Allow-Origin', '*')

    return result


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
