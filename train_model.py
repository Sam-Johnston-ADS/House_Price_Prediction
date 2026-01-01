import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error


# ---------------- LOAD DATA ----------------
df = pd.read_csv(
    r"C:\Users\Sam\Desktop\House_Price_Prediction\data\Bengaluru_House_Data.csv"
)


# ---------------- CLEAN total_sqft ----------------
def convert_sqft_to_num(x):
    try:
        if isinstance(x, str):
            if '-' in x:
                a, b = x.split('-')
                return (float(a) + float(b)) / 2
            return float(x)
    except:
        return None
    return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)


# ---------------- EXTRACT BHK SAFELY ----------------
def extract_bhk(x):
    try:
        if isinstance(x, str):
            return int(x.split()[0])
    except:
        return None
    return None

df['bhk'] = df['size'].apply(extract_bhk)


# ---------------- DROP INVALID ROWS ----------------
df = df.dropna(subset=[
    'total_sqft', 'bath', 'balcony', 'bhk',
    'area_type', 'location', 'price'
])


# ---------------- MODEL DATA ----------------
X = df[['total_sqft', 'bath', 'balcony', 'bhk', 'area_type', 'location']]
y = df['price']

num_features = ['total_sqft', 'bath', 'balcony', 'bhk']
cat_features = ['area_type', 'location']


# ---------------- PIPELINE ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', Ridge(alpha=1.0))
])


# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)


# ---------------- EVALUATE ----------------
y_pred = pipeline.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))


# ---------------- SAVE MODEL ----------------
pickle.dump(pipeline, open("model.pkl", "wb"))
print(" model.pkl saved successfully")

