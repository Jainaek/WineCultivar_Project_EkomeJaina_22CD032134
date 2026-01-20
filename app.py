from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# ===============================
# Load persisted model CONSISTENTLY
# ===============================
with open("model/wine_cultivar_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        input_values = [float(request.form[feature]) for feature in features]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        pred = model.predict(input_scaled)[0]
        prediction = f"Cultivar {pred + 1}"

    return render_template("index.html", features=features, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
