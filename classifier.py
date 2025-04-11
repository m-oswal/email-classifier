import os
import time
import base64
import re
import joblib
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from google_api import create_service
from gmail_api import init_gmail_service, get_email_message_details

# Define all custom classes used in the model


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key."""

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return [item[self.key] for item in data_dict]


class EmailExtractor(BaseEstimator, TransformerMixin):
    """Extract email domains from sender field and one-hot encode them"""

    def __init__(self):
        self.encoder = OneHotEncoder(
            sparse_output=False, handle_unknown='ignore')
        self.domains_seen = set()

    def fit(self, x, y=None):
        # Extract domains and fit the encoder
        domains = self._extract_domains(x)
        self.domains_seen = set(domains)

        # Reshape domains for OneHotEncoder
        domains_array = np.array(domains).reshape(-1, 1)
        self.encoder.fit(domains_array)
        return self

    def transform(self, data):
        # Extract domains
        domains = self._extract_domains(data)

        # Reshape domains for OneHotEncoder
        domains_array = np.array(domains).reshape(-1, 1)

        # Transform to one-hot encoding
        return self.encoder.transform(domains_array)

    def _extract_domains(self, data):
        # Extract email and domain
        email_pattern = r'[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        domains = []

        for text in data:
            match = re.search(email_pattern, str(text))
            domain = match.group(1) if match else "unknown"
            domains.append(domain)

        return domains


class DateExtractor(BaseEstimator, TransformerMixin):
    """Extract date features from emails"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        date_features = []

        for text in data:
            text = str(text)  # Ensure text is string, not numpy array
            # Look for common date patterns
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
                r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY or DD-MM-YYYY
                # DD Mon YYYY
                r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
                # Mon DD, YYYY
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',
                # Weekday, DD Mon YYYY
                r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[a-z]*,?\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
            ]

            # Default values
            has_date = 0
            is_weekend = 0
            hour_of_day = -1

            # Check for each date pattern
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    has_date = 1
                    # More sophisticated date parsing could be done here
                    break

            # Look for time expressions
            time_pattern = r'\b(?:1[0-2]|0?[1-9]):[0-5][0-9]\s*(?:am|pm|AM|PM)?\b|\b(?:2[0-3]|[01]?[0-9]):[0-5][0-9]\b'
            time_match = re.search(time_pattern, text)
            if time_match:
                # Could extract hour information here
                has_time = 1
            else:
                has_time = 0

            # Check for "urgent" time expressions
            urgent_time_patterns = [
                r'\btoday\b', r'\basap\b', r'\bas soon as possible\b',
                r'\bimmediately\b', r'\burgent\b', r'\bdeadline\b',
                r'\bcod\b', r'\bclose of day\b', r'\beod\b', r'\bend of day\b'
            ]

            has_urgent_time = 0
            for pattern in urgent_time_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    has_urgent_time = 1
                    break

            date_features.append([
                has_date,
                has_time,
                has_urgent_time
            ])

        return np.array(date_features)  # Return numpy array


class UrgencyKeywordExtractor(BaseEstimator, TransformerMixin):
    """Extract urgency keywords from text"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        # Keywords for different urgency levels
        urgent_keywords = [
            'urgent', 'immediate', 'asap', 'emergency', 'critical', 'crucial',
            'important', 'priority', 'deadline', 'attention', 'respond', 'now',
            'today', 'quickly', 'action', 'required', 'needed', 'necessary',
            'vital', 'essential', 'alert', 'warning', 'immediately', 'expedite'
        ]

        important_keywords = [
            'important', 'review', 'update', 'please', 'request', 'information',
            'follow-up', 'followup', 'consider', 'attention', 'fyi', 'notification',
            'reminder', 'weekly', 'monthly', 'report'
        ]

        ignore_keywords = [
            'newsletter', 'subscription', 'unsubscribe', 'discount', 'offer',
            'promotion', 'sale', 'digest', 'automated', 'do not reply'
        ]

        features = []
        for text in data:
            text = str(text).lower()  # Ensure text is string, not numpy array

            # Count occurrences of each type of keyword
            urgent_count = sum(
                1 for keyword in urgent_keywords if keyword in text)
            important_count = sum(
                1 for keyword in important_keywords if keyword in text)
            ignore_count = sum(
                1 for keyword in ignore_keywords if keyword in text)

            # Calculate ratios
            total_words = len(text.split())
            urgent_ratio = urgent_count / max(1, total_words)
            important_ratio = important_count / max(1, total_words)
            ignore_ratio = ignore_count / max(1, total_words)

            features.append([
                urgent_count,
                important_count,
                ignore_count,
                urgent_ratio,
                important_ratio,
                ignore_ratio
            ])

        return np.array(features)  # Return numpy array directly


class SubjectFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract specific features from subject lines"""

    def fit(self, x, y=None):
        return self

    def transform(self, subjects):
        features = []

        for subject in subjects:
            subject = str(subject)  # Ensure subject is string, not numpy array

            # Extract features from subject
            has_exclamation = int('!' in subject)
            has_question = int('?' in subject)
            all_caps = int(subject.isupper() and len(subject) > 3)
            length = len(subject)
            word_count = len(subject.split())

            # Check for prefixes like RE:, FWD:
            is_reply = int(bool(re.match(r'^\s*re:', subject, re.IGNORECASE)))
            is_forward = int(
                bool(re.match(r'^\s*(?:fw|fwd):', subject, re.IGNORECASE)))

            # Capitalization patterns
            first_word_caps = 0
            if word_count > 0:
                first_word = subject.split()[0]
                if len(first_word) > 1 and first_word[0].isupper():
                    first_word_caps = 1

            features.append([
                has_exclamation,
                has_question,
                all_caps,
                length,
                word_count,
                is_reply,
                is_forward,
                first_word_caps
            ])

        return np.array(features)  # Return numpy array directly


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Extract length-based features from text"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        features = []

        for text in data:
            text = str(text)  # Ensure text is string, not numpy array

            # Basic length metrics
            char_count = len(text)
            word_count = len(text.split())
            # Avoid division by zero
            sentence_count = max(1, len(re.split(r'[.!?]+', text)) - 1)

            # Calculate averages
            avg_word_length = char_count / max(1, word_count)
            avg_sentence_length = word_count / max(1, sentence_count)

            features.append([
                char_count,
                word_count,
                sentence_count,
                avg_word_length,
                avg_sentence_length
            ])

        return np.array(features)  # Return numpy array directly


# Add the detect_urgency function that was missing
def detect_urgency(subject, body):
    """Detect if an email classified as Important is actually Urgent based on time indicators"""

    # Combine subject and body for analysis
    text = (subject + " " + body).lower()

    # Check for urgent time expressions
    urgent_time_patterns = [
        r'\btoday\b', r'\basap\b', r'\bas soon as possible\b',
        r'\bimmediately\b', r'\burgent\b', r'\bdeadline\b',
        r'\bcod\b', r'\bclose of day\b', r'\beod\b', r'\bend of day\b',
        r'\bdue\s+today\b', r'\boverdue\b', r'\bemergency\b', r'\bcritical\b',
        r'\bimminent\b', r'\bpressing\b', r'\btop\s+priority\b',
        r'\bexpedite\b', r'\bsoonest\b', r'\bprompt\s+attention\b'
    ]

    # Check for date expressions like "tomorrow" or dates within 2 days
    date_urgency_patterns = [
        r'\btomorrow\b', r'\btonight\b', r'\bthis\s+evening\b',
        r'\bin\s+\d+\s+hour', r'\bin\s+the\s+next\s+\d+\s+hour',
        r'\bby\s+end\s+of\s+day\b', r'\bby\s+eod\b', r'\bby\s+cod\b',
        r'\bby\s+noon\b', r'\bby\s+morning\b', r'\bby\s+the\s+end\s+of\b'
    ]

    # Look for time stamps and check if they're within 48 hours
    date_pattern = r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'
    date_matches = re.findall(date_pattern, text)

    # Check for multiple exclamation marks or urgent formatting
    formatting_patterns = [
        r'!!+',  # Multiple exclamation marks
        r'\*urgent\*', r'\*important\*', r'\*priority\*',  # Asterisk emphasis
        r'URGENT', r'ASAP', r'IMMEDIATE',  # All caps urgency
    ]

    # Check for any urgency pattern matches
    for pattern in urgent_time_patterns + date_urgency_patterns + formatting_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Additional logic for dates could be added here
    # For example, parse dates and compare with current date

    return False


# Configuration
CLIENT_SECRET_FILE = 'clientsecret.json'
# Updated to match the new model name
MODEL_FILE = 'binary_email_classifier.pkl'
CHECK_INTERVAL = 60  # Check for new emails every 60 seconds
LABEL_PREFIXES = {
    "No Need to Read": "EmailClassifier/NoNeedToRead",
    "Important": "EmailClassifier/Important",
    "Urgent": "EmailClassifier/Urgent"
}

# Initialize Gmail service


def initialize_services():
    print("Initializing Gmail service...")
    service = init_gmail_service(CLIENT_SECRET_FILE)
    return service

# Create labels if they don't exist


def ensure_labels_exist(service):
    print("Checking for required labels...")
    label_ids = {}

    # Get existing labels
    results = service.users().labels().list(userId='me').execute()
    existing_labels = {label['name']: label['id']
                       for label in results.get('labels', [])}

    # Check and create necessary labels
    for priority, label_name in LABEL_PREFIXES.items():
        if label_name not in existing_labels:
            print(f"Creating label: {label_name}")
            label_data = {
                'name': label_name,
                'messageListVisibility': 'show',
                'labelListVisibility': 'labelShow'
            }
            created_label = service.users().labels().create(
                userId='me', body=label_data).execute()
            label_ids[priority] = created_label['id']
        else:
            print(f"Label already exists: {label_name}")
            label_ids[priority] = existing_labels[label_name]

    return label_ids

# Enhanced function to extract email body


def extract_enhanced_body(payload):
    """Extract the email body from payload with better handling of different formats"""
    body = ""

    # Helper function to extract text from parts recursively
    def extract_from_parts(parts):
        text = ""
        for part in parts:
            if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                text += base64.urlsafe_b64decode(
                    part['body']['data']).decode('utf-8', errors='replace')
            elif part.get('mimeType') == 'text/html' and 'data' in part.get('body', {}):
                html = base64.urlsafe_b64decode(
                    part['body']['data']).decode('utf-8', errors='replace')
                soup = BeautifulSoup(html, 'html.parser')
                text += soup.get_text(separator=' ', strip=True)

            # Recursively check parts
            if 'parts' in part:
                text += extract_from_parts(part['parts'])
        return text

    # Try to get body from different places
    if 'parts' in payload:
        body = extract_from_parts(payload['parts'])
    elif 'body' in payload and 'data' in payload['body']:
        data = payload['body']['data']
        if payload.get('mimeType') == 'text/html':
            html = base64.urlsafe_b64decode(
                data).decode('utf-8', errors='replace')
            soup = BeautifulSoup(html, 'html.parser')
            body = soup.get_text(separator=' ', strip=True)
        else:
            body = base64.urlsafe_b64decode(
                data).decode('utf-8', errors='replace')

    # Clean the body
    if body:
        body = re.sub(r'\s+', ' ', body).strip()

    return body

# Get new unread emails


def get_new_emails(service, history_id=None):
    """Get new unread emails, optionally filtering for those after a specific history ID"""
    query = "is:unread category:primary"

    if history_id:
        # This would be used if Gmail's push notifications were implemented
        # For polling, we'll use timestamps instead
        pass

    try:
        results = service.users().messages().list(
            userId='me', q=query, maxResults=10).execute()
        messages = results.get('messages', [])

        if not messages:
            return []

        email_data = []
        for message in messages:
            msg_id = message['id']
            # Fetch full message
            full_message = service.users().messages().get(
                userId='me', id=msg_id, format='full').execute()

            # Get headers
            headers = full_message['payload'].get('headers', [])
            subject = next((header['value'] for header in headers if header['name'].lower(
            ) == 'subject'), "No Subject")
            sender = next((header['value'] for header in headers if header['name'].lower(
            ) == 'from'), "No Sender")

            # Get body
            body = extract_enhanced_body(full_message['payload'])

            # Store email data
            email_data.append({
                'id': msg_id,
                'thread_id': full_message.get('threadId', ''),
                'sender': sender,
                'subject': subject,
                'body': body,
                'labels': full_message.get('labelIds', [])
            })

        return email_data

    except Exception as e:
        print(f"Error fetching new emails: {e}")
        return []

# Classify email using loaded model - Updated for two-stage approach


def classify_email(model, email_data):
    """Classify a single email using the two-stage approach:
    1. Binary classification (No Need to Read vs Important)
    2. For Important emails, determine if they're urgent
    """
    try:
        # Format the email data for the model
        email_input = [{
            'sender': email_data['sender'],
            'subject': email_data['subject'],
            'body': email_data['body']
        }]

        # Step 1: Binary classification
        binary_prediction = model.predict(email_input)[0]

        # Binary prediction: 0 = No Need to Read, 1 = Important
        if binary_prediction == 0:
            return "No Need to Read"

        # Step 2: For Important emails, check if they're Urgent
        if detect_urgency(email_data['subject'], email_data['body']):
            return "Urgent"
        else:
            return "Important"

    except Exception as e:
        print(f"Error classifying email: {e}")
        # Default to Important if there's an error
        return "Important"

# Apply label to email


def apply_label(service, msg_id, label_id):
    """Apply a label to an email"""
    try:
        service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={'addLabelIds': [label_id]}
        ).execute()
        return True
    except Exception as e:
        print(f"Error applying label to message {msg_id}: {e}")
        return False

# Mark email as read if desired


def mark_as_read(service, msg_id):
    """Mark an email as read by removing the UNREAD label"""
    try:
        service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        return True
    except Exception as e:
        print(f"Error marking message {msg_id} as read: {e}")
        return False

# Log classifications for future reference


def log_classification(email_data, label_text, log_file="email_classifications.csv"):
    """Record email classification for future reference and potential retraining"""
    try:
        # Create new row
        new_row = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sender': email_data['sender'],
            'subject': email_data['subject'],
            'classification': label_text
        }

        # Convert to DataFrame
        df_new = pd.DataFrame([new_row])

        # Append to existing file or create new one
        if os.path.exists(log_file):
            df_existing = pd.read_csv(log_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(log_file, index=False)
        else:
            df_new.to_csv(log_file, index=False)

    except Exception as e:
        print(f"Error logging classification: {e}")


def main():
    print("=== Gmail Email Classification Service ===")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load the classification model
    try:
        print(f"Loading model from {MODEL_FILE}...")
        model = joblib.load(MODEL_FILE)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure the model file exists and is valid")
        return

    # Initialize Gmail service
    service = initialize_services()
    if not service:
        print("Failed to initialize Gmail service")
        return

    # Create or verify labels
    label_ids = ensure_labels_exist(service)
    print("Labels verified and ready")

    # Main service loop
    print(
        f"\nStarting monitoring loop, checking every {CHECK_INTERVAL} seconds")
    print("Press Ctrl+C to stop the service\n")

    try:
        last_run_time = datetime.now()

        while True:
            current_time = datetime.now()
            print(
                f"[{current_time.strftime('%H:%M:%S')}] Checking for new emails...")

            # Get new emails
            new_emails = get_new_emails(service)

            if new_emails:
                print(f"Found {len(new_emails)} new email(s)")

                # Process each email
                for email in new_emails:
                    print(f"Processing: {email['subject']}")

                    # Classify the email using the two-stage approach
                    classification = classify_email(model, email)
                    print(f"  → Classified as: {classification}")

                    # Apply the appropriate label
                    if classification in label_ids:
                        label_applied = apply_label(
                            service, email['id'], label_ids[classification])
                        if label_applied:
                            print(
                                f"  → Applied label: {LABEL_PREFIXES[classification]}")

                    # Log the classification
                    log_classification(email, classification)
            else:
                print("No new emails found")

            # Update last run time
            last_run_time = current_time

            # Sleep until next check
            print(
                f"Next check at {(current_time + timedelta(seconds=CHECK_INTERVAL)).strftime('%H:%M:%S')}")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\nService stopped by user")
    except Exception as e:
        print(f"\nService encountered an error: {e}")
        print("Service will attempt to restart in 60 seconds")
        time.sleep(60)
        main()  # Attempt restart


if __name__ == "__main__":
    main()
