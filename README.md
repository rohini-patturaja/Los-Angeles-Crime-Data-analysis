# 🚨 Los Angeles Crime Dashboard Analysis (2020–Present)


> **Analyzing 1.05 Million Crime Records to Understand When, Where, and Why Crime Happens in Los Angeles**

An interactive data analytics and visualization dashboard that transforms raw LAPD crime data into actionable public safety insights. This project uncovers temporal trends, demographic vulnerabilities, geographic hotspots, and future crime forecasts using advanced data analytics and predictive modeling.

---

## 📊 Project Overview

This project acts like a **weather forecast for public safety**—using historical crime data to anticipate high-risk periods and locations so resources can be deployed proactively rather than reactively.

### Dataset Summary
- **Total Records**: ~1.05 million crime incidents
- **Time Period**: 2020 to Present
- **Features**: 28 columns (temporal, geographic, demographic)
- **Coverage**: All 21 LAPD Community Areas

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Data Processing | Python, Pandas, NumPy |
| Visualization | Plotly, Dash |
| Forecasting | Facebook Prophet |
| Mapping | Folium, HeatMap |
| Analysis | Statistical & Time-Series Methods |

---

## 🎯 Key Insights

### 🕛 The Noon Paradox
- **Peak crime hour**: 12:00 PM (not midnight)
- Crimes align with peak human activity and mobility

---

### 📈 Post-COVID Crime Surge
- Crime increased as lockdowns ended
- **Peak**: 2022 (Central LA)
- **43.4% decrease** from 2023 to 2024 (partial-year data)

---

### ☀️ Seasonal Crime Patterns
- Crime rises in **spring and summer**
- **August** peak
- **December** spike in vehicle burglaries

---

### 🎉 Weekend Risk
- **Friday**: Vehicle burglaries peak
- **Saturday**: Assaults and vandalism increase

---

### 👥 Victim Demographics
- **Most affected age group**: 18–29
- **Gender breakdown**:
  - Male: 55%
  - Female: 35.4% (often domestic violence-related)

---

### 🚗 Vehicle Crime Dominance
- Over **64 vehicle-related crimes per day**
- Vehicle burglary is the **2nd most common crime**

---

### 💻 Digital Crime Rise
- Identity theft ranks among the **top 3 crimes**
- Reflects shift toward cyber-enabled offenses

---

### 🗺️ Crime Hotspots
**High Density Areas**
- Downtown LA
- Hollywood
- South Central

**Low Density Areas**
- Malibu
- Pasadena & Northeast suburbs

---

## 🔮 Predictive Modeling

### 12-Month Crime Forecast
Using **Facebook Prophet**, the model predicts:
- Overall **downward trend**
- Continued seasonal cycles
- Stabilization at reduced crime levels

**Use Case**: Proactive police staffing and resource planning

---

# 🚨 Los Angeles Crime Data Analysis (2020–Present)

> **What if crime patterns were as predictable as the weather?**  
> This project analyzes **1.05 million crime records** from Los Angeles to uncover *when*, *where*, and *why* crime happens — and how data can help prevent it.

Using interactive dashboards, spatial analysis, and time-series forecasting, this project turns raw crime data into **actionable public-safety insights**.

---

## 🧠 The Story Behind the Data

Crime isn’t random.

Just like storms follow seasonal patterns, crime follows **predictable rhythms**:
- Certain hours are riskier than others
- Some neighborhoods consistently face higher exposure
- Specific demographics are disproportionately affected

This project treats crime data like a **forecasting problem**, helping cities prepare *before* incidents spike — not after.

---

## 📊 Dataset Overview

- **Total Records**: ~1.05 million crime incidents  
- **Time Period**: 2020 → Present  
- **Coverage**: All 21 LAPD community areas  
- **Key Features**: Date, time, location, crime type, victim demographics  

The dataset was cleaned, validated, and engineered to ensure reliable insights.

---

## 🛠️ Tech Stack

- **Python** – data processing & modeling  
- **Pandas / NumPy** – data manipulation  
- **Plotly + Dash** – interactive dashboard  
- **Folium** – geographic heatmaps  
- **Facebook Prophet** – time-series forecasting  

---

## 🔍 Key Insights

### 🕛 Crime Peaks at Midday
Contrary to popular belief, **most crimes occur around noon**, not late at night.

**Why it matters**: Patrol strategies focused only on nighttime miss high-risk periods.

---

### 📈 Post-COVID Crime Surge
Crime increased sharply after lockdowns eased, peaking in **2022**, then stabilizing.

---

### ☀️ Strong Seasonal Patterns
Crime rises during warmer months and peaks in **August**, repeating consistently every year.

---

### 🎉 Weekend Effect
- **Fridays** → vehicle burglaries  
- **Saturdays** → assaults & vandalism  

Social activity significantly influences crime patterns.

---

### 👥 Victim Demographics
- Most affected age group: **18–29**
- **55% male**, **35% female** (often domestic violence cases)

---

### 🚗 Vehicle Crimes Dominate
Over **64 vehicle-related crimes per day**, making cars the most targeted asset.

---

### 💻 Rise of Digital Crime
**Identity theft ranks among the top three crimes**, reflecting increasing cyber-enabled offenses.

---

### 🗺️ Geographic Hotspots
High-risk areas include:
- Downtown LA  
- Hollywood  
- South Central  

Lower-risk zones include coastal and suburban regions.

---

## 🔮 Crime Forecasting

Using **Facebook Prophet**, a 12-month forecast reveals:
- A gradual downward trend
- Strong seasonal spikes
- Stabilization if current conditions persist

This supports **proactive policing and resource planning**.

---

## 📈 Dashboard Features

The interactive dashboard enables users to:
- Filter by year and region
- Explore hourly, daily, and seasonal trends
- Visualize crime hotspots via heatmaps
- Analyze victim demographics
- View crime forecasts

Designed for **decision-makers**, not just analysts.

---

## 🧹 Data Cleaning & Preparation

- Removed duplicates and invalid records
- Standardized demographic fields
- Corrected geospatial inconsistencies
- Engineered time-based features

**Result**: >95% complete, high-quality dataset.

---

## 🎯 Strategic Recommendations

Based on insights:
- Reallocate patrols to midday hours
- Prepare seasonal surge responses
- Focus enforcement on hotspot zones
- Strengthen vehicle theft prevention
- Increase identity theft awareness
- Target safety education for young adults

**Projected impact**:
- 20–25% reduction in hotspot crime
- Faster response times
- Improved resource efficiency

---

## 🚀 How to Run the Project

```bash
git clone https://github.com/yourusername/LA-Crime-Analysis.git
cd LA-Crime-Analysis
pip install -r requirements.txt
python Crime_dashboard_app.py

