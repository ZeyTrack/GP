# 🌾 AgroMind – Smart Crop Recommendation & Fertilizer Optimizer

AgroMind is an intelligent web application that assists farmers in making informed agricultural decisions. It recommends the most suitable crop based on soil and environmental data, and applies a linear 

programming model to determine the optimal quantities of N, P, and K fertilizers for maximizing profit while satisfying crop nutrient requirements.


🔍 Problem Statement

Farmers often lack access to scientific guidance regarding:

What crop is best suited for their soil and weather conditions.

How to apply fertilizers in a cost-effective and balanced manner.


✅ Solution

AgroMind solves this by:

Using a Random Forest Classifier to recommend the single most suitable crop.

Applying Linear Programming (LP) to suggest optimal amounts of nitrogen (N), phosphorus (P), and potassium (K) fertilizers to maximize profit while meeting crop-specific nutrient requirements.


📊 Dataset

The dataset includes the following features:

N: Nitrogen content in soil

P: Phosphorus content in soil

K: Potassium content in soil

Temperature (°C)

Humidity (%)

pH: Acidity level of the soil

Rainfall (mm)


🧠 Machine Learning Model

✅ Algorithm Used: Random Forest Classifier

Random Forest was chosen after comparing multiple models due to:

High accuracy and generalization ability

Handles non-linearity and feature interactions well

Robust to outliers and noise


🧮 Fertilizer Optimization (OR Model)

We use Linear Programming (via PuLP) to:

Maximize profit: Profit = Revenue - Fertilizer Costs

Respect nutrient constraints for the predicted crop

Suggest optimal N, P, and K values that fulfill crop needs with minimal cost


🖥️ Streamlit Interface

Users can:

Input their soil and environmental parameters (N, P, K, temperature, humidity, pH, rainfall)

Receive the best crop recommendation

View optimal NPK fertilizer amounts for that crop

🚀 How to Run Locally

After installing all the libraries used :

Run the Streamlit app --->  streamlit run app.py
