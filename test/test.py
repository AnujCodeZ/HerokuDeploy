import requests

resp = requests.post("https://mnist-pytorch.herokuapp.com/predict", files={'file': open("five.jpeg", "rb")})

print(resp.text)