from flask import Flask, render_template, request, jsonify
from model import predict_weak, accuracy

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    study_hours = float(data['study_hours'])
    completion_rate = float(data['completion_rate'])
    score = float(data['score'])

    result = predict_weak([study_hours, completion_rate, score])
    return render_template('index.html', result=result, accuracy=accuracy)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    result = predict_weak([
        data['study_hours'],
        data['completion_rate'],
        data['score']
    ])
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
