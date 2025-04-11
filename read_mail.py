import csv
import re
import base64
import pyperclip as ppc
from bs4 import BeautifulSoup
from gmail_api import init_gmail_service, get_email_messages, get_email_message_details

# Initialize Gmail service
client_file = 'clientsecret.json'
service = init_gmail_service(client_file)

# Fetch emails
messages = get_email_messages(service, max_results=1)

# Prepare CSV file
csv_file = 'emails-test.csv'
fields = ['Sender', 'Subject', 'Body', 'Label']

# Enhanced function to extract email body when the original method fails


def extract_enhanced_body(service, msg_id):
    try:
        # Get the full message to process manually
        full_message = service.users().messages().get(
            userId='me', id=msg_id, format='full').execute()

        # Extract the body from different parts
        payload = full_message.get('payload', {})
        body = ""

        # Helper function to extract text from parts recursively
        def extract_from_parts(parts):
            text = ""
            for part in parts:
                if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                    text += base64.urlsafe_b64decode(
                        part['body']['data']).decode('utf-8')
                elif part.get('mimeType') == 'text/html' and 'data' in part.get('body', {}):
                    html = base64.urlsafe_b64decode(
                        part['body']['data']).decode('utf-8')
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
                html = base64.urlsafe_b64decode(data).decode('utf-8')
                soup = BeautifulSoup(html, 'html.parser')
                body = soup.get_text(separator=' ', strip=True)
            else:
                body = base64.urlsafe_b64decode(data).decode('utf-8')

        return body

    except Exception as e:
        print(f"Error extracting enhanced body for message {msg_id}: {e}")
        return ""

# Clean and sanitize email body with improved handling


def clean_email_body(body):
    if not body:
        return "No content available"

    try:
        # First try to clean HTML
        if '<html' in body.lower() or '<body' in body.lower():
            soup = BeautifulSoup(body, 'html.parser')
            clean_text = soup.get_text(separator=' ', strip=True)
        else:
            # If not HTML, just clean the text
            clean_text = body

        # Remove excessive whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Check if we got anything useful
        if len(clean_text) < 10:
            # Return truncated original if cleaned text is too short
            return body

        return clean_text
    except Exception as e:
        print(f"Error cleaning body: {e}")
        return body[:1000]  # Return truncated original on error

# Special handling for Amazon emails


def is_amazon_email(sender):
    amazon_domains = ['@amazon.', '@amazonaws.', '@marketplace.amazon.']
    return any(domain in sender.lower() for domain in amazon_domains)


# Write to CSV with special handling for Amazon
with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    success_count = 0
    amazon_count = 0

    for msg in messages:
        # Get basic email details using your existing function
        details = get_email_message_details(service, msg['id'])

        sender = details.get('sender', '')
        subject = details.get('subject', '')
        raw_body = details.get('body', '')

        # Special handling for Amazon emails or if body is too short
        is_amazon = is_amazon_email(sender)
        if (is_amazon) or (not raw_body or len(raw_body.strip()) < 10):
            # Try enhanced extraction for Amazon emails
            enhanced_body = extract_enhanced_body(service, msg['id'])
            if enhanced_body and len(enhanced_body) > len(raw_body):
                raw_body = enhanced_body
                print(f"Used enhanced extraction for: {subject}")

        if is_amazon:
            amazon_count += 1
            print(f"Processing Amazon email: {subject}")

        # Clean body
        body = clean_email_body(raw_body)

        # Write to CSV
        if body and body != "No content available":
            writer.writerow({
                'Sender': sender,
                'Subject': subject,
                'Body': body,
                'Label': 'amazon' if is_amazon else ''  # Auto-label Amazon emails
            })
            success_count += 1
        else:
            print(f"Failed to extract content from: {sender} - {subject}")

print(f"Saved {success_count} emails to {csv_file}")
print(f"Processed {amazon_count} Amazon emails")
