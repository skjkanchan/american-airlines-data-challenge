import requests
import pandas as pd
import time


NOAA_TOKEN = ""

airports = ['ABQ', 'AGS', 'ALB', 'AMA', 'ANC', 'ATL', 'AUS', 'AVL', 'BDL', 'BFL',
            'BHM', 'BNA', 'BOI', 'BOS', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE', 'CHS',
            'CID', 'CLE', 'CLT', 'CMH', 'COS', 'CVG', 'DAB', 'DCA', 'DEN', 'DRO',
            'DSM', 'DTW', 'ECP', 'EGE', 'ELP', 'EUG', 'EWR', 'EYW', 'FAT', 'FCA',
            'FLL', 'FSD', 'GEG', 'GRR', 'GSO', 'GSP', 'GUC', 'HDN', 'HNL', 'HSV',
            'IAD', 'IAH', 'ICT', 'ILM', 'IND', 'JAC', 'JAX', 'JFK', 'KOA', 'LAS',
            'LAX', 'LBB', 'LEX', 'LGA', 'LIT', 'MCI', 'MCO', 'MDT', 'MEM', 'MFE',
            'MIA', 'MKE', 'MRY', 'MSN', 'MSO', 'MSP', 'MSY', 'MTJ', 'MYR', 'OGG',
            'OKC', 'OMA', 'ONT', 'ORD', 'ORF', 'PBI', 'PDX', 'PHL', 'PHX', 'PIT',
            'PNS', 'PSP', 'PWM', 'RDM', 'RDU', 'RIC', 'RNO', 'RSW', 'SAN', 'SAT',
            'SAV', 'SBA', 'SBP', 'SDF', 'SEA', 'SFO', 'SJC', 'SJU', 'SLC', 'SMF',
            'SNA', 'SRQ', 'STL', 'STS', 'STT', 'SYR', 'TPA', 'TUL', 'TUS', 'TVC',
            'TYS', 'VPS', 'XNA']

# NOAA uses station IDs in format "GHCND:USW000XXXXX"
# First we need to look up the station ID for each airport
# This maps IATA to NOAA station ID using the LCD dataset

headers = {"token": NOAA_TOKEN}
all_data = []

# Query in chunks of 10 airports (NOAA API limit)
def get_station_id(iata):
    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"
    params = {
        "datatypeid": "HourlyVisibility",
        "locationid": f"AIRPORT:{iata}",
        "limit": 1
    }
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200 and r.json().get("results"):
        return r.json()["results"][0]["id"]
    return None

print("Looking up station IDs...")
station_map = {}
for iata in airports:
    sid = get_station_id(iata)
    if sid:
        station_map[iata] = sid
        print(f"  {iata} → {sid}")
    else:
        print(f"  {iata} → NOT FOUND")
    time.sleep(0.3)  # respect rate limit

print(f"\nFound {len(station_map)} of {len(airports)} stations")

# Save station map for reference
pd.DataFrame(
    list(station_map.items()), columns=["iata", "station_id"]
).to_csv(
    r"C:\Spring 2026\american-airlines-data-challenge\Data\station_map.csv",
    index=False
)
print("Station map saved.")