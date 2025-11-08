🎬 Sentiment Analysis on Movie Reviews

Internship Project — CodeClause | Data Science Domain

🧠 Overview

This project analyzes the sentiment of movie reviews using Natural Language Processing (NLP) techniques.
The model can classify reviews as Positive or Negative, and includes an interactive Streamlit web app where users can input a review or upload a CSV file to analyze multiple reviews at once.

🚀 Features

Clean and minimal UI built with Streamlit

Real-time sentiment prediction for user-inputted reviews

Batch analysis support for uploaded CSV files

Machine Learning pipeline built using TF-IDF + Logistic Regression

Model training, evaluation, and saving using Joblib

Simple visual output and accuracy report

🧩 Project Structure
📁 Sentiment-Analysis-MovieReviews
│
├── 📄 train_model.py          # Script to train and save the model
├── 📄 app.py                  # Streamlit app for the UI
├── 📁 models/                 # Stores trained model and vectorizer
│   ├── model.joblib
│   └── vectorizer.joblib
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md               # Project documentation
└── 📄 dataset.csv             # Movie review dataset

⚙️ Technologies Used

Python 3.10+

Pandas – Data handling

Scikit-learn – Model training and vectorization

NLTK / SpaCy – Text preprocessing

Streamlit – UI for real-time input/output

Joblib – Saving trained models

🧰 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/sentiment-analysis-movie-reviews.git
cd sentiment-analysis-movie-reviews

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # for macOS/Linux
venv\Scripts\activate      # for Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

🧪 Model Training

Run this command to train and save your model:

python train_model.py


This script:

Cleans and preprocesses text data

Trains a sentiment analysis model

Saves the trained model and TF-IDF vectorizer inside the /models folder

🌐 Run the Web App

After training is done, start the Streamlit app:

streamlit run app.py


Then open the displayed local URL (e.g., http://localhost:8501) to access your app.

🖥️ App Usage

🔹 Single Review Mode:
Type a movie review and click Analyze to see whether it’s positive or negative.

🔹 Batch (CSV) Mode:
Upload a CSV file containing a column named review.
The app will analyze all reviews and display:

Predicted sentiment for each row

Probability score

Class distribution chart

📊 Output Example
Review	Sentiment	Probability
“The movie was fantastic!”	Positive	0.94
“It was a complete waste of time.”	Negative	0.89
📚 Learning Outcomes

Text preprocessing using NLP techniques

Feature extraction using TF-IDF

Building and evaluating ML models

Deploying models via Streamlit UI

🧑‍💻 Author

Karthikeyan T
Data Science Intern @ CodeClause
karthick182005@gmail.com
