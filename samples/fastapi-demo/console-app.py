import requests

# Option 1: Using json parameter (automatically sets Content-Type)
url = "http://127.0.0.1:8000/accounts"
data = {
    "name": "John Doe",
    "surname": "Doe",
    "id": 1
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())

# GET all accounts
response = requests.get(url)
print(response.status_code)
print(response.json())

# GET a specific account
account_id = 1
response = requests.get(f"{url}/{account_id}")
print(response.status_code)
print(response.json())

# incorrect post request
data = {
    "name": "Jane Doe",
    "surname": "Doe"
}

response = requests.post(url, json=data)
print(response.status_code)