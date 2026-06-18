import joblib
import pandas as pd

# Load model and feature columns
model = joblib.load("model.pkl")
feature_columns = joblib.load("features.pkl")

print("\n🎬 Netflix Customer Churn Prediction")
print("Enter customer details:\n")

# Create input dictionary with all required features
input_data = {col: 0 for col in feature_columns}

# ---------------- USER INPUT ----------------

input_data["Age"] = int(input("Age: "))
input_data["Subscription_Type"] = int(input("Subscription Type (encoded): "))
input_data["Watch_Hours_Per_Week"] = float(input("Watch Hours Per Week: "))
input_data["Tenure_Months"] = int(input("Tenure (months): "))
input_data["Monthly_Charges"] = float(input("Monthly Charges: "))
input_data["Payment_Delays"] = int(input("Payment Delays: "))
input_data["Customer_Satisfaction"] = float(input("Customer Satisfaction (1-10): "))
input_data["Support_Tickets"] = int(input("Support Tickets: "))
input_data["Genre_Preference"] = int(input("Genre Preference (encoded): "))

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
prediction = model.predict(input_df)

# Output
print("\n========================")
if prediction[0] == 1:
    print("Result: CHURN ❌ (Customer will leave)")
else:
    print("Result: NOT CHURN ✅ (Customer will stay)")
print("========================\n")