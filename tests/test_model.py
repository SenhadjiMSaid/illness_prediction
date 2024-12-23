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


test_sample_data = {
    "Fever": "Yes",
    "Cough": "Yes",
    "Fatigue": "Yes",
    "Difficulty Breathing": "No",
    "Age": 25,
    "Gender": "Male",
    "Blood Pressure": "normal",
    "Cholesterol Level": "normal",
}

test_sample = pd.DataFrame([test_sample_data])

prediction = predict_disease(test_sample)
print(prediction)
