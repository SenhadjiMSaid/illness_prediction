from flask import Flask, render_template, request
import pandas as pd
import joblib

gender_encoder = joblib.load("../models/encoders/gender_encoder.pkl")
disease_encoder = joblib.load("../models/encoders/disease_encoder.pkl")
bp_encoder = joblib.load("../models/encoders/bp_encoder.pkl")
cholesterol_encoder = joblib.load("../models/encoders/cholesterol_encoder.pkl")
scaler = joblib.load("../models/encoders/age_scaler.pkl")
model = joblib.load("../models/best_model/model.pkl")

disease_recommendations = {
    "Asthma": {
        "doctor": "Pulmonologist",
        "advice": "Avoid triggers like smoke and dust, and carry your inhaler at all times.",
    },
    "Bronchitis": {
        "doctor": "Pulmonologist",
        "advice": "Rest, stay hydrated, and avoid smoking or polluted areas.",
    },
    "Chronic Kidney Disease": {
        "doctor": "Nephrologist",
        "advice": "Maintain a low-sodium diet and monitor your kidney function regularly.",
    },
    "Common Cold": {
        "doctor": "General Practitioner",
        "advice": "Rest, drink plenty of fluids, and take over-the-counter medications as needed.",
    },
    "Dengue Fever": {
        "doctor": "Infectious Disease Specialist",
        "advice": "Stay hydrated and monitor your platelet count. Seek immediate care if symptoms worsen.",
    },
    "Diabetes": {
        "doctor": "Endocrinologist",
        "advice": "Monitor your blood sugar levels regularly and maintain a balanced diet.",
    },
    "Eczema": {
        "doctor": "Dermatologist",
        "advice": "Moisturize regularly and avoid harsh soaps or irritants.",
    },
    "Heart Disease": {
        "doctor": "Cardiologist",
        "advice": "Adopt a heart-healthy lifestyle with regular exercise and a balanced diet.",
    },
    "Hypertension": {
        "doctor": "Cardiologist",
        "advice": "Reduce salt intake, exercise regularly, and monitor your blood pressure.",
    },
    "Influenza": {
        "doctor": "General Practitioner",
        "advice": "Rest, stay hydrated, and take antiviral medications if prescribed.",
    },
    "Liver Disease": {
        "doctor": "Hepatologist",
        "advice": "Avoid alcohol and fatty foods. Follow a liver-friendly diet.",
    },
    "Malaria": {
        "doctor": "Infectious Disease Specialist",
        "advice": "Take prescribed antimalarial medications and use mosquito repellents.",
    },
    "Pneumonia": {
        "doctor": "Pulmonologist",
        "advice": "Complete the prescribed antibiotics and get plenty of rest.",
    },
    "Stroke": {
        "doctor": "Neurologist",
        "advice": "Seek immediate medical attention and follow up with rehabilitation therapy.",
    },
    "Tuberculosis": {
        "doctor": "Infectious Disease Specialist",
        "advice": "Complete the full course of medication and ensure proper ventilation.",
    },
}


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

    # Encode Gender
    test_sample["Gender"] = gender_encoder.transform(test_sample["Gender"])

    # One-Hot Encode Blood Pressure
    bp_encoded = bp_encoder.transform(test_sample[["Blood Pressure"]])
    bp_columns = bp_encoder.get_feature_names_out(["Blood Pressure"])
    bp_df = pd.DataFrame(bp_encoded, columns=bp_columns, index=test_sample.index)

    # One-Hot Encode Cholesterol Level
    cholesterol_encoded = cholesterol_encoder.transform(
        test_sample[["Cholesterol Level"]]
    )
    cholesterol_columns = cholesterol_encoder.get_feature_names_out(
        ["Cholesterol Level"]
    )
    cholesterol_df = pd.DataFrame(
        cholesterol_encoded, columns=cholesterol_columns, index=test_sample.index
    )

    # Drop original columns and concatenate encoded columns
    test_sample = test_sample.drop(columns=["Blood Pressure", "Cholesterol Level"])
    test_sample = pd.concat([test_sample, bp_df, cholesterol_df], axis=1)

    # Align with training columns
    training_columns = model.feature_names_in_
    for col in training_columns:
        if col not in test_sample:
            test_sample[col] = 0
    test_sample = test_sample[training_columns]

    # Scale Age
    test_sample[["Age"]] = scaler.transform(test_sample[["Age"]])

    # Predict and decode the disease
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
    print(symptoms)

    all_symptoms = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
    symptom_data = {
        symptom: "Yes" if symptom in symptoms else "No" for symptom in all_symptoms
    }

    # Check for normal values
    if blood_pressure == "Normal" and cholesterol_level == "Normal" and not symptoms:
        likely_not_sick = True
    else:
        likely_not_sick = False

    test_sample_data = {
        **symptom_data,
        "Age": int(age),
        "Gender": gender,
        "Blood Pressure": blood_pressure,
        "Cholesterol Level": cholesterol_level,
    }
    print("AAAAAAAAAAAAAAAAAA")
    test_sample = pd.DataFrame([test_sample_data])
    print(test_sample_data)

    # Only predict if not all values are normal
    prediction = predict_disease(test_sample) if not likely_not_sick else "No disease"

    recommendation = disease_recommendations.get(prediction, {})
    response = {
        "Age": age,
        "Gender": gender,
        "Symptoms": symptoms,
        "Blood Pressure": blood_pressure,
        "Cholesterol Level": cholesterol_level,
        "prediction": prediction,
        "doctor": recommendation.get("doctor", "General Practitioner"),
        "advice": recommendation.get(
            "advice", "Monitor your health and maintain a healthy lifestyle."
        ),
        "likely_not_sick": likely_not_sick,
    }
    return render_template("response.html", results=response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
