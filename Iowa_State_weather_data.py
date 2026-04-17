import requests
import pandas as pd
import time
import os

OUTPUT_DIR = r"C:\Spring 2026\american-airlines-data-challenge\Data\weather-NOAA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

all_dfs = []
failed = []

for iata in airports:
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={iata}"
        f"&data=vsby&data=sknt&data=skyc1&data=skyl1&data=wxcodes"
        f"&year1=2025&month1=1&day1=1"
        f"&year2=2025&month2=12&day2=31"
        f"&tz=UTC&format=comma&latlon=no&missing=M&trace=T&direct=no&report_type=3"
    )
    
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.text) > 200:
            lines = r.text.strip().split("\n")
            # Skip comment lines starting with #
            data_lines = [l for l in lines if not l.startswith("#")]
            from io import StringIO
            df = pd.read_csv(StringIO("\n".join(data_lines)))
            all_dfs.append(df)
            print(f"  {iata} ✓ ({len(df)} rows)")
        else:
            print(f"  {iata} ✗ (empty response)")
            failed.append(iata)
    except Exception as e:
        print(f"  {iata} ✗ ({e})")
        failed.append(iata)
    
    time.sleep(0.5)  # be polite to the server

# Combine all airports
print(f"\nCombining {len(all_dfs)} airport datasets...")
metar_2025 = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows: {len(metar_2025)}")
print(f"Columns: {metar_2025.columns.tolist()}")
print(f"Failed airports: {failed}")

# Save
out_path = os.path.join(OUTPUT_DIR, "metar_2025.csv")
metar_2025.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print(metar_2025.head(3).to_string())