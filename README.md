#  Los Angeles Crime Intelligence Dashboard  
### Turning 1.05 Million Crime Records Into a Safer City

> *What if we could predict crime the way meteorologists predict weather — not to frighten people, but to prepare them?*

---

##  The Problem

Los Angeles is one of the most dynamic cities in the world — home to over 4 million residents across 21 community areas. Every day, the LAPD generates massive volumes of incident data, but historically, this data has been underutilized.

This project answers a critical question:

> **What patterns are hidden inside 1.05 million crime records, and how can they be used to prevent crime instead of reacting to it?**

This dashboard enables **proactive policing** — helping decision-makers deploy resources *before* crime surges occur.

---

##  Dataset Overview

| Attribute            | Details                          |
|---------------------|----------------------------------|
| Total Records       | ~1,050,000 crime incidents       |
| Time Span           | January 2020 – Present           |
| Features            | 28 columns per record            |
| Geographic Coverage | All 21 LAPD Community Areas      |
| Data Source         | LAPD Open Data Portal            |

The dataset includes:
- Time of occurrence (hour, day, month, season)
- Victim demographics
- Crime categories
- Geographic coordinates

---

##  Tech Stack

| Layer            | Tools & Technologies                  |
|------------------|--------------------------------------|
| Data Processing  | Python, Pandas, NumPy               |
| Visualization    | Plotly, Dash                        |
| Forecasting      | Facebook Prophet                    |
| Mapping          | Folium, HeatMap                     |
| Analysis         | Statistical & Time-Series Methods   |

---

##  Data Cleaning & Feature Engineering

To ensure reliability:
- Removed duplicate records  
- Fixed geospatial inconsistencies  
- Standardized demographic categories  
- Engineered time-based features (hour, weekday, season)

 Result: **95%+ complete dataset with high integrity**

---

##  Key Insights

### 1.  Crime Peaks at Noon — Not Midnight
- Highest crime occurs at **12:00 PM**
- Driven by increased daytime activity

 **Implication:** Patrol strategies must shift toward midday coverage

---

### 2. COVID Impact & Post-Pandemic Surge
- Crime dropped during lockdowns  
- **2022 peak** followed by a **43.4% decline (2023–2024)**

---

### 3.  Crime is Seasonal
- Peaks: **Spring & Summer (August highest)**
- December spike in **vehicle burglaries**

 **Implication:** Seasonal resource planning is critical

---

### 4.  Weekly Crime Patterns
- **Friday:** Vehicle burglaries peak  
- **Saturday:** Assault & vandalism increase  

---

### 5.  Most Affected Demographic
- **Ages 18–29** are most impacted  
- Gender distribution:
  - Male: 55%  
  - Female: 35.4% (higher domestic violence cases)

---

### 6.  Vehicle Crime Epidemic
- **64+ vehicle crimes daily**
- Second most common crime type

---

### 7.  Rise of Identity Theft
- Now among **top 3 crime types**
- Indicates shift toward **cyber-enabled crime**

---

### 8.  Geographic Concentration
High-risk zones:
- Downtown LA  
- Hollywood  
- South Central  

Lower-risk zones:
- Malibu  
- Pasadena  

**Implication:** Targeted policing > uniform distribution

---

##  12-Month Forecast (Prophet Model)

Key predictions:
-  Continued downward trend from 2022 peak  
-  Persistent seasonal cycles  
-  Stabilization at lower crime levels  

---

##  Explore the Dashboard

🔗 **[View the Live Tableau Dashboard](https://public.tableau.com/app/profile/rohini.patturaja/viz/LACrimeAnalysis_17749157248070/LACrimeAnalysisduring2020-2025)**

> Interactive features:
- Filter by year & region  
- Analyze hourly & seasonal trends  
- Explore crime heatmaps  
- View forecast projections  

---

##  Recommendations

1. Rebalance patrols toward **midday hours**
2. Implement **summer surge staffing**
3. Focus resources on **high-density hotspots**
4. Launch **vehicle theft awareness campaigns**
5. Expand **cybercrime capabilities**
6. Develop **youth-focused safety programs**

---

## Expected Impact

- 20–25% reduction in hotspot crime  
- Faster response times  
- Improved resource allocation efficiency  

---

##  Conclusion

Los Angeles is not unmanageable — it is **complex**.

This project demonstrates how large-scale data:
- Reveals hidden crime patterns  
- Enables proactive decision-making  
- Transforms policing from reactive → predictive  

> **Insight-driven policing can change outcomes — before crime happens.**

---

##  Built With

- Python, Plotly, Dash  
- Facebook Prophet  
- LAPD Open Data Portal  

---

##  Author

**Rohini Patturaja**  
MS Data Science & AI | Data Analyst  

🔗 [LinkedIn](https://linkedin.com/in/rohini-patturaja)

---

© 2025 Rohini Patturaja
