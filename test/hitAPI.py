import requests

url = "http://localhost:5000/predict"

files = {'file': open('test\personal_transactions.csv', 'rb')}
data = {'days': 7}
response = requests.post(url, files=files, data=data)
print(response.json())
