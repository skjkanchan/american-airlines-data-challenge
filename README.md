# ✈️ Airline Crew Sequences Meet Bad Weather
### EPPS-American Airlines Data Analytics Challenge — GROW 26.2

---

## Overview

Airlines frequently face disruptions caused by weather, delays, and operational constraints. Certain airports are more prone to weather-related delays than others, particularly during specific seasons of the year. One of the most impactful ways to minimize these disruptions is through smarter pilot scheduling — specifically, deciding which flights should be assigned to the same pilot in a combination called a **sequence**.

Sequences that pair high-risk flights together can lead to:
- Cascading delays across multiple flights
- Missed connections due to tight turnarounds
- Duty time violations
- Increased pilot fatigue and operational risk

---

## Problem Statement

The goal of this challenge is to **identify pairs of airports that should not appear together in the same pilot sequence**, in order to reduce disruptions and improve overall reliability.

For this challenge, we focus on a simplified sequence structure:

```
Airport A → DFW → Airport B
```

Where DFW is American Airlines' largest hub, and A and B can be any airport in the United States.

### Example

> Airports **A** and **B** are both prone to sudden thunderstorms in spring, while airport **C** is not. Assigning a pilot to fly **A → DFW → B** is therefore more likely to produce delays than **A → DFW → C** or **B → DFW → C**. The goal is to flag pairings like (A, B) as sub-optimal so schedulers can avoid them in future sequences.

---

## Data Sources

| Source | Description | Link |
|--------|-------------|------|
| FAA ASPM | Airport-level flight performance metrics, scheduling patterns, and NAS delay rates | https://aspm.faa.gov |
| BTS On-Time Performance | Flight-level delay records, weather delay codes, and tail numbers for crew chain reconstruction | https://www.transtats.bts.gov |
| Iowa State ASOS Archive | Historical METAR observations including visibility, wind speed, ceiling height, and precipitation type for 2025 | https://mesonet.agron.iastate.edu |
| Aviation Weather Center | Aviation-specific weather products and real-time METAR cache | https://aviationweather.gov |
| NOAA | Historical and seasonal weather data for correlating meteorological conditions with delay patterns | https://www.weather.gov |
| FAR Part 117 (FAA) | Regulatory duty time limits and rest minimums encoded as fixed constraints | Hardcoded thresholds — no download required |

---

## Approach

Our model identifies high-risk airport pairings by combining three core dimensions:

1. **Operational Flight Performance** — Historical delay patterns and airport throughput metrics from FAA ASPM and BTS
2. **Environmental Risk** — Seasonal and real-time weather conditions from NOAA and AWC
3. **Employee Safety Restrictions** — FAR Part 117 duty time and fatigue constraints encoded directly into the feature pipeline

---

## Repository Structure

```
├── Data/
│   ├── Monthly Data/           # BTS On-Time Performance CSVs (Jan–Dec 2025)
│   ├── BTS Yearly/             # BTS annual delay cause breakdown
│   └── weather-AWC/
│       ├── raw/                # Raw METAR cache files
│       └── processed/          # Cleaned weather data (weather_all_airports_clean.csv)
├── weather-NOAA/
│   └── metar_2025.csv          # Iowa State ASOS historical METAR data
├── embedding_model.py          # Main model training and data preparation pipeline
├── visualize_embeddings.py     # PCA embedding visualization and risk analysis
├── generate_pairings.py        # Inference script to score all airport pairs and generate ranked avoid list
├── risky_pairings.csv          # Ranked avoid list — 15,006 airport combinations scored by cascade probability
├── airport_features.csv        # Cleaned ASPM airport statistics
├── DFW_sequences_enriched.csv  # Final enriched sequence dataset
├── embedding_model.pth         # Trained model weights
├── airport_label_encoder.npy   # Airport label encoder classes
├── airport_embeddings.png      # Embedding visualization output
├── training_loss.png           # Training loss curve
└── README.md
```

---

## Competition Details

- **Hosted by:** UTD EPPS & American Airlines
