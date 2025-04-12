from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

# Gmail API Scope (Read Emails)
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    """Authenticate and return Gmail service."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def test_gmail():
    """Test fetching the latest 10 emails."""
    try:
        service = get_gmail_service()
        results = service.users().messages().list(userId='me', maxResults=10).execute()
        messages = results.get('messages', [])

        if not messages:
            print("✅ Gmail API is working, but no emails found.")
        else:
            print("✅ Gmail API is working! Here are the latest emails:")
            for msg in messages:
                print(f"Email ID: {msg['id']}")
    except Exception as e:
        print("❌ Gmail API Error:", e)

# Run test
test_gmail()
