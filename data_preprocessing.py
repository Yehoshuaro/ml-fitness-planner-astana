import pandas as pd

def load_datasets(user_path: str, ref_path: str):
    df_user = pd.read_csv(user_path)
    df_ref = pd.read_excel(ref_path)

    rename_map = {
        'What is your age?': 'age',
        'What is your gender?': 'sex',
        'What is your height in cm? [Number]': 'height_cm',
        'What is your weight in kg? [Number]': 'weight_kg',
        'What is your top fitness goal/-s?': 'goal',
        'Any health conditions that an AI app should consider for your plans?': 'health',
    }

    df_user = df_user.rename(columns=rename_map)
    df_user = df_user[list(rename_map.values())].copy()

    for col in df_user.select_dtypes(include='object').columns:
        df_user[col] = df_user[col].astype(str).str.strip()

    def parse_age(x):
        digits = ''.join(ch if ch.isdigit() else ' ' for ch in str(x)).split()
        return int(digits[0]) if digits else None

    df_user['age'] = df_user['age'].apply(parse_age)
    df_user['height_cm'] = pd.to_numeric(df_user['height_cm'], errors='coerce')
    df_user['weight_kg'] = pd.to_numeric(df_user['weight_kg'], errors='coerce')
    df_user['bmi'] = df_user['weight_kg'] / ((df_user['height_cm'] / 100) ** 2)

    ref_map = {
        'Sex': 'sex',
        'Age': 'age',
        'Height': 'height_cm',
        'Weight': 'weight_kg',
        'BMI': 'bmi',
        'Fitness Goal': 'goal',
        'Exercises': 'exercises',
        'Diet': 'diet',
        'Recommendation': 'recommendation'
    }

    df_ref = df_ref.rename(columns=ref_map)
    df_ref = df_ref[list(ref_map.values())].copy()

    for col in df_ref.select_dtypes(include='object').columns:
        df_ref[col] = df_ref[col].astype(str).str.strip()

    return df_user, df_ref
