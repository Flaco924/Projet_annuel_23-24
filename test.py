import os

credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if credentials_path:
    print(f"GOOGLE_APPLICATION_CREDENTIALS is set to: {credentials_path}")
    if os.path.exists(credentials_path):
        print("Credentials file is accessible.")
    else:
        print("Credentials file is not accessible.")
else:
    print("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
