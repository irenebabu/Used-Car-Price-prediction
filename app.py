import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
ct = joblib.load("transformer.pkl")
scaler = joblib.load("scaler.pkl")


df = pd.read_csv("cars_24_combined.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)
df.dropna(subset=["Car Name", "Year"], inplace=True)
df["Location"] = df["Location"].fillna("Unknown")

st.set_page_config(page_title="Used Car Price Predictor")

st.title("🚗 Used Car Price Predictor")

car_name = st.selectbox(
    "Car Name",
    options=["Select Car"] + sorted(df["Car Name"].unique().tolist())
)

year = st.number_input(
    "Manufacturing Year",
    min_value=2000,
    max_value=2025,
    value=2020
)

distance = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=10000
)

owner = st.selectbox(
    "Number of Owners",
    sorted(df["Owner"].unique())
)

fuel = st.selectbox(
    "Fuel Type",
    sorted(df["Fuel"].unique())
)

state_map = {
    "KL": "Kerala",
    "TN": "Tamil Nadu",
    "KA": "Karnataka",
    "MH": "Maharashtra",
    "DL": "Delhi",
    "HR": "Haryana",
    "UP": "Uttar Pradesh",
    "TS": "Telangana",
    "AP": "Andhra Pradesh",
    "RJ": "Rajasthan",
    "PB": "Punjab",
    "GJ": "Gujarat",
    "WB": "West Bengal",
    "MP": "Madhya Pradesh",
    "CG": "Chhattisgarh",
    "BR": "Bihar",
    "JH": "Jharkhand",
    "OD": "Odisha",
    "AS": "Assam",
    "UK": "Uttarakhand"
}

locations = sorted([loc for loc in df["Location"].unique() if loc != "Unknown"])


display_locations = {}

for loc in locations:
    state_code = loc.split("-")[0]

    state_name = state_map.get(
        state_code,
        "Unknown State"
    )

    display_locations[
        f"{loc} ({state_name})"
    ] = loc


selected_display = st.selectbox(
    "Location",
    list(display_locations.keys())
)

location = display_locations[selected_display]


drive = st.selectbox(
    "Transmission",
    sorted(df["Drive"].unique())
)

car_type = st.selectbox(
    "Vehicle Type",
    sorted(df["Type"].unique())
)


st.subheader("Selected Vehicle")

st.info(f"""
🚘 Car: {car_name}

📅 Year: {year}

🛣️ Distance: {distance:,} km

⛽ Fuel: {fuel}

👤 Owners: {owner}
""")
if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Car Name": [car_name],
        "Year": [year],
        "Distance": [distance],
        "Owner": [owner],
        "Fuel": [fuel],
        "Location": [location],
        "Drive": [drive],
        "Type": [car_type]
    })
    input_encoded = ct.transform(input_df)

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    st.success(
        f"Estimated Price: ₹{prediction:,.0f}"
    )