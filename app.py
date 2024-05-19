from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model_path = 'spam_classifier_model.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'No message field provided in request'}), 400
    message = data['message']
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

