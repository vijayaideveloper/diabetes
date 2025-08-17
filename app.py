from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    feature1 = float(request.form["feature1"])
    feature2 = float(request.form["feature2"])
    feature3 = float(request.form["feature3"])
    feature4 = float(request.form["feature4"])
    feature5 = float(request.form["feature5"])
    feature6 = float(request.form["feature6"])
    feature7 = float(request.form["feature7"])
    feature8 = float(request.form["feature8"])
    final_features = np.array([[feature1], [feature2], [feature3],
                               [feature4], [feature5], [feature6],
                               [feature7], [feature8]]).reshape(1, -1)

    prediction = model.predict(final_features)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return render_template("index.html", prediction_text=f"Prediction: {result}")


if __name__ == "__main__":
    app.run(debug=True)
