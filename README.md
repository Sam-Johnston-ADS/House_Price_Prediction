# 🏠 House Price Prediction using Machine Learning

This project predicts house prices based on various features such as area, number of bedrooms, bathrooms, balconies, availability, and location.  
It uses Machine Learning with a complete preprocessing and training pipeline to ensure accurate predictions.

---

## 📌 Project Overview

House price prediction is a real-world problem in real estate analytics.  
This project focuses on cleaning real-world data, handling mixed data formats, and building a robust machine learning pipeline for prediction.

---

## 🧠 Technologies Used

- Programming Language: Python 3.12  
- Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
- Model: Linear Regression  
- Model Saving: Joblib

---

## 📂 Project Structure
House_Price_Prediction/
│
├── data/
│ └── Bengaluru_House_Data.csv
│
├── train_model.py
├── predict.py
├── model.pkl
├── README.md
└── requirements.txt

---

## 🔍 Features Used

- Total Square Feet
- BHK (Bedrooms)
- Bathrooms
- Balconies
- Area Type
- Availability
- Location

---

## ⚙️ Data Preprocessing

- Removed missing and invalid values
- Converted `size` column into numeric BHK
- Cleaned `total_sqft` values (handled ranges)
- One-hot encoded categorical features
- Scaled numerical features
- Used Pipeline and ColumnTransformer

---

## 🏗️ Model Training

- Train-test split applied
- Linear Regression model trained
- Performance metrics used:
  - R² Score
  - Mean Absolute Error (MAE)
- Trained model saved as `model.pkl`

---

## 📈 Model Performance

| Metric | Value |
|------|------|
| R² Score | 0.509 |
| MAE | 35.79 Lakhs |

---

## 🔮 Sample Prediction
Input:
Area: 1200 sq.ft
BHK: 2
Bathrooms: 2
Balcony: 1
Location: Whitefield

Predicted House Price:
₹71.54 Lakhs


---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```
python train_model.py
```
### 3. Predict House Price
```
python predict.py
```

## 🎯 Use Cases

Real estate price estimation

Machine learning academic project

Data preprocessing pipeline practice

## 🚀 Future Enhancements

Use advanced ML models

Add EDA notebook

Build a Streamlit web app

Improve accuracy with hyperparameter tuning

## 👨‍💻 Author

Sam Johnston C
B.Tech – Artificial Intelligence & Data Science
St. Joseph College of Engineering



