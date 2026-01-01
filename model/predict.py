import pickle
import pandas as pd


# Load trained pipeline
model = pickle.load(open("model.pkl", "rb"))


def predict_price(total_sqft, bath, balcony, bhk, area_type, location):

    input_df = pd.DataFrame([{
        'total_sqft': total_sqft,
        'bath': bath,
        'balcony': balcony,
        'bhk': bhk,
        'area_type': area_type,
        'location': location
    }])

    price = model.predict(input_df)[0]
    return round(price, 2)


# ---------------- RUN ----------------
if __name__ == "__main__":
    price = predict_price(
        total_sqft=1200,
        bath=2,
        balcony=1,
        bhk=2,
        area_type="Super built-up  Area",
        location="Whitefield"
    )

    print(f"Predicted House Price: ₹{price} Lakhs")
