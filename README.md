# 📧 Email Classification System using Machine Learning and Gmail API

This project is an intelligent email classification system that automatically categorizes incoming emails into three labels:

- **No Need to Read**
- **Important**
- **Urgent**

It uses a combination of supervised machine learning (binary classification) and rule-based logic for urgency detection. The model is integrated with the **Gmail API** to fetch emails, classify them, and apply labels directly in your inbox.

---

## 🚀 Features

- Fetches unread emails using Gmail API
- NLP-based feature extraction from subject, sender, and body
- Machine learning classifier to filter non-relevant emails
- Rule-based urgency detection for deadline-sensitive messages
- Applies labels automatically: `NoNeedToRead`, `Important`, `Urgent`
- Logging of classifications for analysis

---

## 🗂 File Structure

| File                | Description                                       |
|---------------------|---------------------------------------------------|
| `classifier.py`     | Main script for classification and automation     |
| `read_mail.py`      | Handles data reading and preprocessing            |
| `gmail_api.py`      | Functions for Gmail API authentication and access |
| `google_api.py`     | Wrapper to initialize the Gmail API service       |
| `binary_email_classifier.pkl` | Trained ML model for binary classification     |

---

## 📦 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/email-classifier.git
   cd email-classifier
