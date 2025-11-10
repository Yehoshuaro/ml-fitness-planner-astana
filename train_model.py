import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH = "data/"
USER_FILE = os.path.join(DATA_PATH, "AI_Personalized_Fitness_App_Ответы_Ответы_на_форму_1.csv")
REF_FILE = os.path.join(DATA_PATH, "gym_recommendation.xlsx")
GYM_FILE = os.path.join(DATA_PATH, "megaGymDataset.csv")

print("Loading datasets...")
df_user = pd.read_csv(USER_FILE)
df_ref = pd.read_excel(REF_FILE)
df_gym = pd.read_csv(GYM_FILE)
print("Datasets loaded successfully!")

def sex_to_num(x):
    return 1 if str(x).lower().startswith('m') else 0

def map_goal(text):
    t = str(text).lower()
    if 'loss' in t: return 'weight_loss'
    if 'gain' in t or 'muscle' in t: return 'muscle_gain'
    if 'endurance' in t: return 'endurance'
    return 'general'

def train_classifier(df_ref, df_user, df_gym):
    print("Preparing data...")

    for df in [df_ref, df_user, df_gym]:
        if 'goal' in df.columns:
            df['label'] = df['goal'].apply(map_goal)
        else:
            df['label'] = 'general'

    df = pd.concat([df_ref, df_user, df_gym], ignore_index=True)

    needed_cols = ['sex', 'age', 'height_cm', 'weight_kg', 'bmi']
    for col in needed_cols:
        if col not in df.columns:
            df[col] = -1

    df = df.dropna(subset=['age', 'label'])

    X = pd.DataFrame({
        'sex_male': df['sex'].apply(sex_to_num),
        'age': df['age'],
        'height_cm': df['height_cm'],
        'weight_kg': df['weight_kg'],
        'bmi': df['bmi']
    }).fillna(-1)

    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/classifier.joblib")

    print("Model trained and saved to model/classifier.joblib")

    return clf, X_test, y_test

if __name__ == "__main__":
    clf, X_test, y_test = train_classifier(df_ref, df_user, df_gym)
    print("Done. Model ready for predictions")
