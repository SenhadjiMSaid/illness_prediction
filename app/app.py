from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    age = request.form.get("age")
    gender = request.form.get("gender")
    symptoms = request.form.getlist("symptoms")  # Get multiple checkbox values
    blood_pressure = request.form.get("blood-pressure")
    cholesterol_level = request.form.get("cholesterol")

    # Placeholder response (replace with ML model prediction)
    response = {
        "Age": age,
        "Gender": gender,
        "Symptoms": symptoms,
        "Blood Pressure": blood_pressure,
        "Cholesterol Level": cholesterol_level,
    }
    print(response)
    return render_template("response.html", results=response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
