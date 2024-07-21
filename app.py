from flask import Flask, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')


# Load the model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]  # Remove stopwords and stem
    return ' '.join(words)

@app.route('/')
def home():
    return "Flask is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    title = data.get('title', '')
    text = data.get('text', '')
    subject = data.get('subject', '')
    combined_text = f"{title} {text} {subject}"
    processed_text = preprocess_text(combined_text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)
    result = "Fake" if prediction[0] == 0 else "Real"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
