from flask import Flask, render_template, request
import pandas as pd
import joblib

gender_encoder = joblib.load("../models/encoders/gender_encoder.pkl")
disease_encoder = joblib.load("../models/encoders/disease_encoder.pkl")
bp_encoder = joblib.load("../models/encoders/bp_encoder.pkl")
cholesterol_encoder = joblib.load("../models/encoders/cholesterol_encoder.pkl")
scaler = joblib.load("../models/encoders/age_scaler.pkl")
model = joblib.load("../models/best_model/model.pkl")


def predict_disease(test_sample):
    """
    Predict the disease based on the provided test sample.

    Parameters:
    test_sample (pd.DataFrame): A single-row DataFrame containing the test sample.

    Returns:
    str: The decoded disease prediction.
    """
    binary_columns = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
    for col in binary_columns:
        test_sample[col] = test_sample[col].map({"Yes": 1, "No": 0})

    test_sample["Gender"] = gender_encoder.transform(test_sample["Gender"])

    bp_encoded = bp_encoder.transform(test_sample[["Blood Pressure"]])
    bp_columns = bp_encoder.get_feature_names_out(["Blood Pressure"])
    bp_df = pd.DataFrame(bp_encoded, columns=bp_columns, index=test_sample.index)

    cholesterol_encoded = cholesterol_encoder.transform(
        test_sample[["Cholesterol Level"]]
    )
    cholesterol_columns = cholesterol_encoder.get_feature_names_out(
        ["Cholesterol Level"]
    )
    cholesterol_df = pd.DataFrame(
        cholesterol_encoded, columns=cholesterol_columns, index=test_sample.index
    )

    test_sample = test_sample.drop(columns=["Blood Pressure", "Cholesterol Level"])
    test_sample = pd.concat([test_sample, bp_df, cholesterol_df], axis=1)

    training_columns = model.feature_names_in_
    for col in training_columns:
        if col not in test_sample:
            test_sample[col] = 0
    test_sample = test_sample[training_columns]

    test_sample[["Age"]] = scaler.transform(test_sample[["Age"]])

    encoded_prediction = model.predict(test_sample)

    decoded_prediction = disease_encoder.inverse_transform(
        encoded_prediction.astype(int)
    )

    return decoded_prediction[0]


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    age = request.form.get("age")
    gender = request.form.get("gender")
    symptoms = request.form.getlist("symptoms")
    blood_pressure = request.form.get("blood-pressure")
    cholesterol_level = request.form.get("cholesterol")

    all_symptoms = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
    symptom_data = {
        symptom: "Yes" if symptom in symptoms else "No" for symptom in all_symptoms
    }

    test_sample_data = {
        **symptom_data,
        "Age": int(age),
        "Gender": gender,
        "Blood Pressure": blood_pressure,
        "Cholesterol Level": cholesterol_level,
    }
    test_sample = pd.DataFrame([test_sample_data])
    prediction = predict_disease(test_sample)
    print(prediction)

    response = {
        "Age": age,
        "Gender": gender,
        "Symptoms": symptoms,
        "Blood Pressure": blood_pressure,
        "Cholesterol Level": cholesterol_level,
        "prediction": prediction,
    }
    # print(response)
    return render_template("response.html", results=response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
