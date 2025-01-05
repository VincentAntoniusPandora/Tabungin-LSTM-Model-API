from google.oauth2 import service_account
from google.auth.transport.requests import Request
import requests

url = "https://getprediction-497063330583.asia-southeast1.run.app"
KEY_PATH = 'test/tabungin-lstm-prediction-815643e1cc37.json'

# Authenticate and get access token
scopes = ['https://www.googleapis.com/auth/cloud-platform']
creds = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=scopes)
creds.refresh(Request())
access_token = creds.token

headers = {
    'Authorization': f'Bearer {access_token}'
}

files = {'file': open('test\personal_transactions.csv', 'rb')}
data = {'days': 7}
response = requests.post(url, files=files, data=data, headers=headers)
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}")
