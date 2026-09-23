# 📱 Used Phone Price Prediction

A Machine Learning project built with Python to predict the resale price of used smartphones using a dataset of 1,000,000 records. The model evaluates multiple features such as device specifications, physical condition, market demand, and seller background to accurately estimate resale values[cite: 1].

---

## 📌 Project Overview

With the secondary smartphone market growing rapidly, setting an accurate resale price is crucial for both sellers and buyers. This project processes large-scale data, performs feature engineering/encoding, and leverages a **Random Forest Regressor** to predict phone resale prices with an impressive **R² score of ~0.96**[cite: 1].

---

## 📊 Dataset Overview

* **Dataset Size:** 1,000,000 rows × 28 columns[cite: 1]
* **Target Variable:** `resale_price`[cite: 1]

### Feature Categories:
* **Device Specifications:** `brand`, `model`, `release_year`, `ram_gb`, `storage_gb`, `screen_size_inches`, `battery_capacity`, `processor_score`, `camera_score`, `os_type`, `has_5g`[cite: 1]
* **Usage & History:** `original_price`, `purchase_year`, `age_months`, `usage_hours_per_day`, `battery_health`, `repair_history`[cite: 1]
* **Physical Condition:** `condition`, `screen_cracked`, `body_damage`, `water_damage`[cite: 1]
* **Market & Accessories:** `city_tier`, `seller_type`, `warranty_remaining_months`, `box_available`, `charger_available`, `market_demand_score`[cite: 1]

---

## 🛠 Project Workflow

1. **Data Loading & Inspection:**
   * Loaded `used_phone_price_prediction_1M.csv` using Pandas[cite: 1].
   * Verified structure, column data types, missing values (0 nulls), and duplicate records (0 duplicates)[cite: 1].

2. **Data Preprocessing & Feature Encoding:**
   * Dropped redundant columns (`brand`)[cite: 1].
   * Applied **One-Hot Encoding** (`pd.get_dummies`) to categorical features (`model`, `os_type`, `condition`, `city_tier`, `seller_type`) with `drop_first=True`[cite: 1].

3. **Model Training:**
   * Train-Test Split: **80% Training / 20% Testing** (`random_state=42`)[cite: 1].
   * Model: **Random Forest Regressor** (`n_estimators=100`, `max_depth=20`, `n_jobs=-1`)[cite: 1].

4. **Evaluation:**
   * Computed MAE, MSE, RMSE, and R² score metrics[cite: 1].

---

## 📈 Model Performance Results

The tuned Random Forest Regressor achieved the following results on the 200,000 test samples:

| Metric | Score |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | `2,499.90`[cite: 1] |
| **Mean Squared Error (MSE)** | `10,898,356.59`[cite: 1] |
| **Root Mean Squared Error (RMSE)** | `3,301.27`[cite: 1] |
| **R² Score** | **`0.9610` (96.1% Variance Explained)**[cite: 1] |

---

## 🚀 How to Run

### Prerequisites
Install the required Python packages:

```bash
pip install pandas numpy scikit-learn

---

├── used_phone_price_prediction_1M.csv   # Dataset (1 Million records)
├── used phone price prediction.ipynb    # Jupyter Notebook with ML code
└── README.md                            # Project documentation
