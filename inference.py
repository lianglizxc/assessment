import requests


image_file_path = "./data/sample_test/54.jpg"

url = "http://127.0.0.1:1234/predict/"

with open(image_file_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

prediction = response.json()['prediction']
# example prediction
print(prediction)