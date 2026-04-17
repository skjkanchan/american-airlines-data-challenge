import requests

NOAA_TOKEN = ""
headers = {"token": NOAA_TOKEN}

# Test 1: confirm token works
r = requests.get(
    "https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets",
    headers=headers
)
print("Token test status:", r.status_code)

# Test 2: search for DFW station directly by name
r2 = requests.get(
    "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations",
    headers=headers,
    params={
        "datatypeid": "HourlyVisibility",
        "limit": 5,
        "sortfield": "name",
        "name": "Dallas Fort Worth"
    }
)
print("DFW search status:", r2.status_code)
print(r2.json())