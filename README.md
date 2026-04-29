# CMPSC-445---Best-Restaurants-in-PA
# Pennsylvania Restaurant Finder

**CMPSC 445 Final Project** – *Best Restaurants in Pennsylvania*  
**Team:** Savannah Gamage & Mekhi Saunders  
**Date:** April 28th, 2026  

---

## Overview

Pennsylvania Restaurant Finder is a machine learning-powered web application that predicts restaurant quality scores using an **XGBoost regression model**.

Instead of relying solely on raw user ratings, the model incorporates:

- Cuisine type (Mexican, Italian, Fast Food, etc.)
- Price level ($ – $$$$)
- Review volume and recency
- Sentiment analysis of reviews
- Social media presence (Instagram followers)
- Table service tags (takeout, delivery, outdoor seating, etc.)

Users can filter restaurants by **city** (or "All of Pennsylvania") and **cuisine type**. The app returns a ranked list of the top 10 restaurants based on `predicted_rating`.

The interface is clean, responsive, and works on desktop, tablet, and mobile.

---

## Live Demo (Video)

[https://pennstateoffice365-my.sharepoint.com/:v:/g/personal/slg6111_psu_edu/IQBwUedEyBYtQrmtXWXNyIahAVUzkf2ScPVvF-XmzMpv7j8?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=jISGUa
]

---

## Repository Structure

```
.
├── app.py
├── restaurant_finder.html
├── PA_restaurants_model.py
├── best_restaurant_pipeline.pkl
├── model_features.pkl
├── restaurants_pa_clean.csv
├── restaurants_pa_model_ready.csv
├── restaurants_with_predictions.csv
├── requirements.txt
├── actual_vs_predicted_best.png
├── residuals_best.png
├── feature_importance_best.png
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pa-restaurant-finder.git
cd pa-restaurant-finder
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Backend Server

```bash
python app.py
```

Expected output:

```
Starting server...
API documentation available at http://localhost:8000/docs
Root endpoint: http://localhost:8000/
```

### 4. Open the Web Application

Go to:

```
http://localhost:8000/
```

API docs:

```
http://localhost:8000/docs
```

---

## How It Works

### Model Training (`PA_restaurants_model.py`)

- Loads `restaurants_pa_model_ready.csv` (408 records, 91 features)
- Performs feature selection (top 15 non-categorical features)
- Trains:
  - Linear Regression
  - Random Forest
  - XGBoost
- Tunes XGBoost using 5-fold GridSearchCV
- Saves best model as `best_restaurant_pipeline.pkl`

---

### Backend (`app.py`)

- FastAPI server
- Loads trained pipeline
- Endpoint: `/restaurants`

**Query Parameters:**
- `city`
- `cuisine`
- `top_n` (default = 10)

Returns JSON sorted by `predicted_rating`.

---

### Frontend (`restaurant_finder.html`)

- Pure HTML/CSS/JavaScript
- No build step required
- Loads restaurant data once on startup
- Features:
  - City dropdown (includes "All of Pennsylvania")
  - Cuisine grouping (Popular vs Suggested)
  - Dynamic ranked results display

---

## Example API Call

```
GET http://localhost:8000/restaurants?city=Pittsburgh&cuisine=mexican&top_n=5
```

Returns:

- JSON list of restaurants
- Sorted by `predicted_rating` (descending)

---

## Model Performance (XGBoost)

| Metric        | Value      |
|--------------|-----------|
| Test R²       | 0.5834    |
| RMSE          | 0.829     |
| MAE           | 0.455     |
| CV Mean R²    | 0.8238    |
| CV Std R²     | 0.0602    |

**Key Insights:**

- Predictions are within ~0.5 stars on average
- Most important features:
  - `review_recency_ratio`
  - `log_reviews_count_360d`
  - `sentiment_avg`
  - `price_level`
- Residuals are centered at zero (no major bias)


## Data Collection

- Source: Beamstation dataset (~4,000 restaurants)
- Due to cost (> $1,000), used:
  - 16 free samples from:
    - GitHub
    - HuggingFace
    - Beamstation samples

Final dataset:

- **408 unique restaurants**
- **125 columns**

### Final Files:

- `restaurants_pa_clean.csv` → full dataset
- `restaurants_pa_model_ready.csv` → training features (91 columns)
- `restaurants_with_predictions.csv` → includes `predicted_rating`

---

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- FastAPI
- HTML / CSS / JavaScript
- Joblib
- Matplotlib / Seaborn

---

## Team Contributions

- **Savannah Gamage**
  - Data collection (16 samples)
  - Preprocessing
  - Feature engineering
  - Frontend design
  - Demo Recording

- **Mekhi Saunders**
  - Model training
  - Hyperparameter tuning
  - FastAPI backend
  - Integration
  - Report writing

We both contributed to debugging, testing, and documentation.


## License

Educational use only (CMPSC 445 – Penn State)

---

## Acknowledgments

- Beamstation (dataset samples)
- GitHub & HuggingFace (public data)
- Scikit-learn, XGBoost, FastAPI communities
