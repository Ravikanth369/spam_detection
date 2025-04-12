import os
import pickle
import base64
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load trained model
model = load_model("model/spam_model.keras")

# Load TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Gmail API Scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Log for spam prediction history
spam_logs = []

# Prediction stats counter
prediction_stats = defaultdict(int)

# In-memory storage for spam over time graph
spam_over_time_counts = defaultdict(int)

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def authenticate_user(email):
    creds = None
    token_path = f'tokens/{email}_token.json'

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            os.makedirs('tokens', exist_ok=True)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def fetch_recent_emails(service, email_id):
    """Fetch the most recent emails for the provided email ID"""
    results = service.users().messages().list(userId=email_id, maxResults=10).execute()
    messages = results.get('messages', [])

    email_texts = []
    for msg in messages:
        msg_data = service.users().messages().get(userId=email_id, id=msg['id']).execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])

        subject, sender = "", ""
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            if header['name'] == 'From':
                sender = header['value']

        email_body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    email_body = base64.urlsafe_b64decode(part['body']['data']).decode("utf-8", errors='ignore')
                    break
        elif 'body' in payload and 'data' in payload['body']:
            email_body = base64.urlsafe_b64decode(payload['body']['data']).decode("utf-8", errors='ignore')

        if email_body.strip():
            email_texts.append({
                "subject": subject,
                "sender": sender,
                "content": email_body
            })

    return email_texts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan_emails', methods=['GET'])
def scan_emails():
    email_id = request.args.get('email', 'me')  # Get the email to scan
    try:
        # Authenticate the user for the specified email
        service = authenticate_user(email_id)  # Use the function to authenticate for each email

        # Fetch emails for the specific user
        emails = fetch_recent_emails(service, email_id)  

        if not emails:
            return jsonify({"status": "No emails found for the given email address."})

        results = []
        for email in emails:
            email_features = vectorizer.transform([email["content"]]).toarray()
            prediction_prob = model.predict(email_features)[0][0]
            prediction = "Spam" if prediction_prob > 0.4 else "Normal"
            prediction_stats[prediction] += 1

            today = datetime.now().strftime("%Y-%m-%d")
            spam_logs.append({"date": today, "prediction": prediction})
            if prediction == "Spam":
                spam_over_time_counts[today] += 1

            results.append({
                "subject": email["subject"],
                "sender": email["sender"],
                "prediction": prediction,
                "accuracy": f"{prediction_prob * 100:.2f}%"
            })

        return jsonify(results)

    except Exception as e:
        print(f"Error: {str(e)}")  # Add detailed logging for debugging
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    email_features = vectorizer.transform([email_text]).toarray()
    prediction_prob = model.predict(email_features)[0][0]
    prediction = "Spam" if prediction_prob > 0.4 else "Normal"

    prediction_stats[prediction] += 1
    today = datetime.now().strftime("%Y-%m-%d")
    spam_logs.append({"date": today, "prediction": prediction})
    if prediction == "Spam":
        spam_over_time_counts[today] += 1

    return jsonify({
        "prediction": prediction,
        "accuracy": f"{prediction_prob * 100:.2f}%"
    })

@app.route('/spam_over_time')
def spam_over_time():
    try:
        if not spam_over_time_counts:
            return jsonify({"labels": ["No Data"], "data": [0]})

        sorted_data = sorted(spam_over_time_counts.items())[-10:]
        labels = [item[0] for item in sorted_data]
        data = [item[1] for item in sorted_data]

        return jsonify({"labels": labels, "data": data})
    except Exception as e:
        print("Error fetching spam data:", e)
        return jsonify({"labels": ["Error"], "data": [0]})

@app.route('/reports')
def reports():
    labels = ['Spam', 'Normal']
    values = [prediction_stats['Spam'], prediction_stats['Normal']]
    if sum(values) == 0:
        values = [1, 1]

    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['red', 'green'])
    plt.title('Spam Detection Statistics')
    plt.savefig('static/report.png')

    return render_template('reports.html', report_image='static/report.png')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
