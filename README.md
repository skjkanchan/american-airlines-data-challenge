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
| Aviation Weather Center | Historical METAR observations including ceiling height, visibility, wind speed, and precipitation type | https://aviationweather.gov |
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
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory data analysis and modeling notebooks
├── src/                # Feature engineering and model pipeline
├── outputs/            # Results, predictions, and visualizations
├── report/             # Final team report (PDF)
└── README.md
```

---

## Competition Details

- **Hosted by:** UTD EPPS & American Airlines
- **Submission Deadline:** April 17, 2025 by 5:00 PM
- **Format:** Team report (Word or PDF, max 20 pages)
- **Contact:** Jose Ramirez-Hernandez — Jose.Ramirez-hernandez@aa.com
