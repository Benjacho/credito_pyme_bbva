from flask import Flask, request, jsonify
import pickle
from model_files.ml_model import predict_model
app = Flask('pymes')


@app.route('/api/predict', methods=['POST'])
def predict():
    credit = request.get_json()
    print(credit)

    with open('./model_files/model.sav', 'rbb' as f_in):
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_model(credit, model)

    result = {
        'predictions': list(predictions)
    }

    return result


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
