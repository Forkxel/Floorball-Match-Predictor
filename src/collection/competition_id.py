import requests

API_KEY = "MY_API_KEY"
ACCESS_LEVEL = "trial"
LANGUAGE_CODE = "en"
FORMAT = "json"

url = f"https://api.sportradar.com/floorball/{ACCESS_LEVEL}/v2/{LANGUAGE_CODE}/competitions.{FORMAT}"
response = requests.get(url, params={"api_key": API_KEY}, timeout=30)
response.raise_for_status()

data = response.json()

for competition in data.get("competitions", []):
    name = competition.get("name")
    cid = competition.get("id")
    gender = competition.get("gender")
    print(cid, "|", name, "|", gender)