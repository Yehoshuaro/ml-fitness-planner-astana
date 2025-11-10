import joblib
import pandas as pd

clf = joblib.load("model/classifier.joblib")

sample_user = pd.DataFrame([{
    "sex_male": 1,
    "age": 20,
    "height_cm": 184,
    "weight_kg": 73,
    "bmi": 21.6
}])

prediction = clf.predict(sample_user)[0]

print(f"Recommended fitness goal: {prediction}")
