# Fake-News-Detection-FlaskApp
Fake News Detection Web App

OverView
The Fake News Detection project aims to classify news articles as either fake or real using machine learning techniques. This project addresses the widespread issue of misinformation by leveraging Natural Language Processing (NLP) to analyze and predict the authenticity of news content.

Dataset
The dataset used is the ISOT Fake News Dataset, comprising two CSV files:

True.csv: Contains over 12,600 articles from reliable news sources like Reuters.
Fake.csv: Contains over 12,600 articles from unreliable sources flagged by organizations like Politifact and Wikipedia.
Steps Involved
Data Loading and Preprocessing:

Load the dataset.
Clean and preprocess the text data, including removing stopwords and stemming.
Feature Extraction:

Use TfidfVectorizer to convert text data into numerical features.
Model Training:

Split the data into training and testing sets.
Train a machine learning model (e.g., Logistic Regression) on the training data.
Model Evaluation:

Evaluate the model using accuracy, precision, recall, and F1-score.
Test the model with sample texts to validate its performance.
Deployment (Optional):

Save the trained model using joblib.
Deploy the model using Flask for a simple web interface.
Usage
Training the Model:

Run the provided Jupyter Notebook or Google Colab script to preprocess data, train the model, and evaluate its performance.
Testing the Model:

Use the predict_fake_news function to test the model with new text samples.
Deployment:

(Optional) Deploy the model using Flask for a web-based prediction service.
Dependencies
Python 3.x
Pandas
Numpy
Scikit-learn
NLTK
Flask (for deployment)
Joblib
Conclusion
This project demonstrates the application of machine learning and NLP in detecting fake news, offering a valuable tool in combating misinformation.

