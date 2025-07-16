import streamlit as st
import pandas as pd
import joblib
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value

# --- Load ML Model ---
model = joblib.load('best_random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="AgroMind Fertilizer Optimizer", layout="centered")

st.title("🌾 AgroMind ")
st.write("Enter soil and weather data:")

# --- Input fields for crop prediction ---
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0)
P_input = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=43.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.2)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# --- Crop economic info inputs ---
st.subheader("📦 Economic Parameters")
Y = st.number_input("Enter expected crop yield (kg per hectare):", min_value=0.0, value=30.0)
P_price = st.number_input("Enter selling price of crop (EGP per kg):", min_value=0.0, value=20.0)

# --- Fertilizer Costs ---
st.subheader("💵 Fertilizer Costs (EGP per kg):")
fertilizer_names = ["Urea", "SSP", "Potassium Sulphate", "Power Grow"]
default_costs = {"Urea": 70, "SSP": 58, "Potassium Sulphate": 89, "Power Grow": 57}
costs = {}
for f in fertilizer_names:
    costs[f] = st.number_input(f"{f}:", min_value=0.0, value=float(default_costs[f]))

# --- Predict and Optimize Button ---
if st.button("🔍 Recommend & Optimize"):
    input_df = pd.DataFrame([[N, P_input, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    pred_encoded = model.predict(input_df)[0]
    selected_crop = label_encoder.inverse_transform([pred_encoded])[0]

    st.success(f"✅ Recommended Crop: **{selected_crop}**")

    # --- Optimization Function ---
    def optimize_fertilizer(selected_crop, Y, P, costs):
        crop_nutrient_data = {
            "rice": ((80, 100), (40, 50), (30, 40)),
            "maize": ((100, 150), (50, 60), (40, 50)),
            "chickpea": ((20, 25), (40, 50), (30, 40)),
            "kidney beans": ((25, 30), (50, 60), (30, 40)),
            "pigeonpeas": ((25, 30), (30, 40), (30, 40)),
            "mothbeans": ((20, 25), (30, 40), (20, 30)),
            "mungbean": ((20, 30), (30, 40), (20, 30)),
            "blackgram": ((20, 30), (30, 40), (20, 30)),
            "lentil": ((25, 30), (40, 50), (30, 40)),
            "watermelon": ((80, 120), (40, 60), (60, 100)),
            "muskmelon": ((80, 120), (40, 60), (60, 100)),
            "apple": ((50, 80), (30, 50), (30, 50)),
            "orange": ((100, 150), (30, 60), (200, 300)),
            "papaya": ((100, 150), (50, 75), (150, 200)),
            "coconut": ((150, 200), (50, 75), (250, 300)),
            "cotton": ((150, 200), (50, 80), (100, 150)),
            "jute": ((100, 150), (50, 60), (100, 120)),
            "coffee": ((150, 200), (50, 75), (150, 200)),
            "pomegranate": ((40, 60), (30, 40), (40, 50)),
            "banana": ((200, 250), (100, 150), (200, 250)),
            "mango": ((150, 200), (50, 75), (150, 200)),
            "grapes": ((50, 100), (25, 50), (50, 100)),
        }

        fertilizers = {
            "Urea": {"N": 0.46, "P": 0.00, "K": 0.00},
            "SSP": {"N": 0.00, "P": 0.08, "K": 0.00},
            "Potassium Sulphate": {"N": 0.00, "P": 0.00, "K": 0.50},
            "Power Grow": {"N": 0.19, "P": 0.19, "K": 0.19}
        }

        if selected_crop not in crop_nutrient_data:
            return None, None, "❌ Crop nutrient data not found."

        (N_range, P_range, K_range) = crop_nutrient_data[selected_crop]
        N_min, _ = N_range
        P_min, _ = P_range
        K_min, _ = K_range

        model = LpProblem("Fertilizer_Optimization_Model", LpMaximize)
        x = {f: LpVariable(f"x_{f}", lowBound=0) for f in fertilizers}

        revenue = Y * P
        total_cost = lpSum(costs[f] * x[f] for f in fertilizers)
        model += revenue - total_cost, "Total_Profit"

        model += lpSum(fertilizers[f]["N"] * x[f] for f in fertilizers) >= N_min
        model += lpSum(fertilizers[f]["P"] * x[f] for f in fertilizers) >= P_min
        model += lpSum(fertilizers[f]["K"] * x[f] for f in fertilizers) >= K_min

        model.solve()

        if model.status == 1:
            plan = {f: x[f].varValue for f in fertilizers if x[f].varValue and x[f].varValue > 0.001}
            profit = value(model.objective)
            return profit, plan, "✅ Optimization successful."
        else:
            return None, None, "❌ No optimal solution found."

    # --- Run Optimization ---
    profit, plan, status_msg = optimize_fertilizer(selected_crop, Y, P_price, costs)
    if plan:
        st.subheader("📈 Optimized Fertilizer Plan")
        st.success(status_msg)
        st.write(f"💰 **Estimated Profit**: {profit:.2f} EGP")
        st.markdown("**Fertilizer Usage (kg):**")
        for fert, qty in plan.items():
            st.write(f"- {fert}: {qty:.2f} kg")
    else:
        st.error(status_msg)
