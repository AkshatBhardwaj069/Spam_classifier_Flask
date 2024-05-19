from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model_path = r'C:\Users\vishakhag\Downloads\email_classifier\spam_classifier_model.joblib'
vectorizer_path = r'C:\Users\vishakhag\Downloads\email_classifier\tfidf_vectorizer.joblib'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file not found. Please ensure both files are in the specified paths.")

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
    app.run(debug=True)

