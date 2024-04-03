import requests
import json

ec2_public_ip_address = '54.92.160.213'
inference_server_port = 8080

url = 'http://' + ec2_public_ip_address + ':' + str(inference_server_port) + '/generate'

data = {
    "inputs": """
        <Human>: I can't seem to feel any emotion except anxiety, not even for myself."
        <AI>: 
    """,
    "parameters": {"max_new_tokens": 100}
}

headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=data, headers=headers)

print(response.status_code)
print(response.text)
