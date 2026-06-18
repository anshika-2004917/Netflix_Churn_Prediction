from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/churn_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    watch_hours = int(request.form["watch_hours"])
    tenure = int(request.form["tenure"])
    monthly_charges = int(request.form["monthly_charges"])
    payment_delays = int(request.form["payment_delays"])
    satisfaction = int(request.form["satisfaction"])
    support_tickets = int(request.form["support_tickets"])

    data = np.array([[
        age,
        0,
        watch_hours,
        tenure,
        monthly_charges,
        payment_delays,
        satisfaction,
        support_tickets,
        0
    ]])

    prediction = model.predict(data)[0]

    result = "Customer Will Churn" if prediction == 1 else "Customer Will Stay"

    return render_template(
        "result.html",
        prediction=result
    )

if __name__ == "__main__":
    app.run(debug=True)