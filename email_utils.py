import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM")

def send_email(subject: str, body: str, to_email: str):
    msg = MIMEMultipart()
    msg['From'] = SMTP_FROM
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()  # Identify the server
            server.starttls()  # Secure the connection
            server.ehlo()  # Re-identify after starting TLS
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, to_email, msg.as_string())
            print("Email sent successfully!")
    except smtplib.SMTPAuthenticationError as e:
        print("SMTP Authentication Error:", e)
    except Exception as e:
        print(f"Error sending email: {e}")
