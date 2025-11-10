import joblib
import pandas as pd
import os
import math

CLF_PATH = os.path.join("model", "classifier.joblib")
clf = None
if os.path.exists(CLF_PATH):
    clf = joblib.load(CLF_PATH)

DATA_PATH = "data"
GYM_FILE = os.path.join(DATA_PATH, "megaGymDataset.csv")
REF_FILE = os.path.join(DATA_PATH, "gym_recommendation.xlsx")

df_gym = pd.DataFrame()
df_ref = pd.DataFrame()
if os.path.exists(GYM_FILE):
    df_gym = pd.read_csv(GYM_FILE).fillna("")
if os.path.exists(REF_FILE):
    try:
        df_ref = pd.read_excel(REF_FILE).fillna("")
    except Exception:
        df_ref = pd.DataFrame()

def safe_float(x, default=None):
    try:
        return float(x)
    except:
        return default

def sex_to_num_semi(x):
    return 1 if str(x).lower().startswith("m") else 0

def calc_bmi(weight, height_cm):
    try:
        h = float(height_cm)/100.0
        return round(float(weight) / (h*h), 1)
    except Exception:
        return None

def budget_to_tier(budget_kzt):
    if budget_kzt is None: return "unknown"
    s = str(budget_kzt).replace(",", "").replace(" ", "")
    try:
        if "-" in s:
            b = float(s.split("-")[-1])
        else:
            b = float(s)
    except:
        return "unknown"
    if b >= 30000: return "expensive"
    if 10000 <= b < 30000: return "mid"
    if 1 <= b < 10000: return "budget"
    return "unknown"

def choose_training_setting(budget_tier, age, schedule_pref):
    if int(age) < 16:
        return {"where":"home_or_outdoor", "reason":"Under 16 -> home/outdoor recommended"}
    if budget_tier == "expensive":
        return {"where":"gym", "reason":"budget allows gym"}
    if budget_tier == "mid":
        return {"where":"gym_or_home", "reason":"mid budget -> flexible"}
    if budget_tier == "budget":
        return {"where":"home_or_outdoor", "reason":"budget-limited -> home/outdoor"}
    return {"where":"home_or_gym", "reason":"unknown budget"}

def choose_duration_by_time(schedule_level):
    if schedule_level <= 1: return 40
    if schedule_level <= 3: return 40
    return 60

def build_workout_template(goal_label, availability_per_week, health_conditions, equipment_list, existing_sports):

    workouts = []
    avail = int(max(1, availability_per_week))
    health = ("" if health_conditions is None else str(health_conditions).lower())
    existing = [e.lower() for e in (existing_sports or [])]

    if "rehab" in goal_label or "rehabil" in health:
        for i in range(avail):
            workouts.append({
                "day_index": i,
                "focus": "rehabilitation / mobility",
                "session_minutes": 30,
                "exercises": [
                    {"name":"controlled hip hinge (no heavy load)", "sets":"3","reps":"8-12"},
                    {"name":"isometric core activation", "sets":"3","reps":"20s"},
                    {"name":"banded glute bridges", "sets":"3","reps":"12-15"},
                    {"name":"shoulder mobility + scapular drills", "sets":"3","reps":"8-12"}
                ]
            })
        return workouts

    if "asthma" in health or "breath" in health or "астм" in health:
        cardio_type = "stationary bike / indoor cycling / elliptical"
    else:
        cardio_type = "running / cycling / brisk walk"

    if "muscle" in goal_label or "gain" in goal_label:
        if avail <= 2:
            for i in range(avail):
                workouts.append({
                    "day_index": i,
                    "focus":"full_body_strength",
                    "session_minutes": choose_duration_by_time(avail),
                    "exercises":[
                        {"name":"squat variation (bodyweight or goblet)","sets":"3","reps":"6-12"},
                        {"name":"deadlift variant or hip hinge (light)","sets":"3","reps":"6-10"},
                        {"name":"horizontal push (push-up/bench)","sets":"3","reps":"6-12"},
                        {"name":"pull (row/lat pull)","sets":"3","reps":"6-12"},
                        {"name":"core stability","sets":"3","reps":"20-30s"}
                    ]
                })
        elif avail == 3:
            workouts = [
                {"day_index":0,"focus":"upper_push_pull","session_minutes":50,"exercises":[{"name":"bench/press","sets":"3","reps":"6-12"},{"name":"rows","sets":"3","reps":"6-12"},{"name":"accessory arms","sets":"3","reps":"8-15"}]},
                {"day_index":1,"focus":"lower_strength","session_minutes":50,"exercises":[{"name":"squat/variation","sets":"4","reps":"6-10"},{"name":"deadlift/hinge","sets":"3","reps":"5-8"},{"name":"calves/core","sets":"3","reps":"12-20"}]},
                {"day_index":2,"focus":"full_body_hypertrophy","session_minutes":45,"exercises":[{"name":"compound circuits","sets":"3","reps":"8-15"},{"name":"isolation","sets":"3","reps":"10-15"}]}
            ]
        else:
            workouts = [
                {"day_index":0,"focus":"push","session_minutes":60,"exercises":[{"name":"bench/overhead press","sets":"4","reps":"6-10"},{"name":"dips/pushups","sets":"3","reps":"8-15"}]},
                {"day_index":1,"focus":"pull","session_minutes":60,"exercises":[{"name":"barbell/rows","sets":"4","reps":"6-10"},{"name":"pullups/lat pulldown","sets":"3","reps":"6-12"}]},
                {"day_index":2,"focus":"legs","session_minutes":60,"exercises":[{"name":"squat","sets":"4","reps":"6-10"},{"name":"romanian deadlift","sets":"3","reps":"6-10"}]},
                {"day_index":3,"focus":"conditioning_or_accessory","session_minutes":40,"exercises":[{"name":cardio_type,"sets":"1","reps":"20-30min"},{"name":"mobility","sets":"1","reps":"10-15min"}]}
            ]
    elif "weight_loss" in goal_label or "lose" in goal_label:
        for i in range(avail):
            workouts.append({
                "day_index": i,
                "focus":"cardio + light strength",
                "session_minutes": choose_duration_by_time(avail),
                "exercises": [
                    {"name":cardio_type, "sets":"1", "reps":"25-40min"},
                    {"name":"circuit bodyweight strength", "sets":"3", "reps":"12-20"}
                ]
            })
    elif "endurance" in goal_label:
        for i in range(avail):
            workouts.append({
                "day_index": i,
                "focus":"endurance",
                "session_minutes": 40 if avail<=3 else 60,
                "exercises":[{"name":"steady-state cardio","sets":"1","reps":"30-60min"},{"name":"mobility/core","sets":"2","reps":"10-20min"}]
            })
    else:
        for i in range(avail):
            workouts.append({
                "day_index": i,
                "focus":"general_fitness",
                "session_minutes": choose_duration_by_time(avail),
                "exercises":[
                    {"name":"mixed circuit (push/pull/legs)","sets":"3","reps":"8-15"},
                    {"name":"cardio short","sets":"1","reps":"15-25min"}
                ]
            })

    if "knee" in health or "колен" in health:
        for w in workouts:
            for ex in w["exercises"]:
                ex["note"] = ex.get("note","")
                if "squat" in ex["name"].lower() or "jump" in ex["name"].lower():
                    ex["note"] += " ОГРАНИЧЕНИЕ: избегать глубоких приседов/прыжков; заменить на hip hinge / leg press low load."

    if "back" in health or "спин" in health or "hernia" in health:
        for w in workouts:
            for ex in w["exercises"]:
                ex["note"] = ex.get("note","")
                if "deadlift" in ex["name"].lower() or "hinge" in ex["name"].lower():
                    ex["note"] += " ОГРАНИЧЕНИЕ: работать с малым весом, акцент на технику, заменить на hip-hinge без прогиба."

    return workouts

def generate_diet_recommendation(goal_label, weight_kg, age, activity_level, budget_tier):

    try:
        kcal_baseline = float(weight_kg) * 24
    except:
        kcal_baseline = 2000
    act_mul = 1.2
    if activity_level >= 4: act_mul = 1.5
    kcal_total = int(kcal_baseline * act_mul)

    diet = {"estimated_daily_kcal": kcal_total, "macros": {}}
    if "muscle" in goal_label:
        diet["macros"] = {"protein_g": int(2.0*float(weight_kg)), "carbs_g":"moderate-to-high", "fat_g":"moderate"}
        diet["advice"] = "Увеличить калорийность на +200-500 kcal/день, белок ~2.0 г/кг, распределять по 3-5 приёмам пищи."
    elif "weight_loss" in goal_label:
        diet["macros"] = {"protein_g": int(1.6*float(weight_kg)), "carbs_g":"reduced", "fat_g":"moderate"}
        diet["advice"] = "Создать дефицит ~300-500 kcal/день, белок 1.6–2.0 г/кг, увеличить NEAT и кардио."
    else:
        diet["macros"] = {"protein_g": int(1.4*float(weight_kg)), "carbs_g":"balanced", "fat_g":"balanced"}
        diet["advice"] = "Поддерживающий режим, 3–4 приёма пищи, держать белок ~1.4 г/кг."
    # бюджетные советы
    if budget_tier == "budget":
        diet["budget_advice"] = "Использовать яйца, творог, консервированную рыбу, бобовые — дешёвые источники белка."
    elif budget_tier == "mid":
        diet["budget_advice"] = "Добавить птицу, сезонные овощи, крупы."
    else:
        diet["budget_advice"] = "Можешь позволить более разнообразный рацион и протеиновые добавки при необходимости."

    return diet

def generate_plan(user_dict: dict):

    age = int(user_dict.get("age", 25))
    sex = user_dict.get("sex", user_dict.get("sex_male", "Male"))
    height = safe_float(user_dict.get("height_cm") or user_dict.get("height") or 170, 170)
    weight = safe_float(user_dict.get("weight_kg") or user_dict.get("weight") or 70, 70)
    bmi = calc_bmi(weight, height)
    sessions = int(user_dict.get("sessions_per_week") or user_dict.get("desired_sessions") or 3)
    budget_kzt = user_dict.get("budget_kzt") or user_dict.get("budget") or "0-5000"
    health_conditions = user_dict.get("health_conditions") or user_dict.get("health") or ""
    existing_activities = user_dict.get("existing_activities") or user_dict.get("current_activities") or []
    if isinstance(existing_activities, str):
        existing_activities = [x.strip() for x in existing_activities.split(",") if x.strip()]

    goal_user_text = (user_dict.get("goal") or "").lower()

    budget_tier = budget_to_tier(budget_kzt)

    predicted_label = None
    if clf is not None:
        try:
            feat = pd.DataFrame([{
                "sex_male": sex_to_num_semi(sex),
                "age": age,
                "height_cm": height,
                "weight_kg": weight,
                "bmi": bmi or -1
            }])
            predicted_label = clf.predict(feat)[0]
        except Exception:
            predicted_label = None

    if goal_user_text:
        if "muscl" in goal_user_text or "gain" in goal_user_text:
            label = "muscle_gain"
        elif "lose" in goal_user_text or "weight loss" in goal_user_text or "fat" in goal_user_text:
            label = "weight_loss"
        elif "endurance" in goal_user_text:
            label = "endurance"
        elif "rehab" in goal_user_text or "rehabil" in goal_user_text or "recover" in goal_user_text:
            label = "rehab"
        else:
            label = predicted_label or "general"
    else:
        label = predicted_label or "general"

    workouts = build_workout_template(label, sessions, health_conditions, equipment_list=None, existing_sports=existing_activities)

    activity_level = int(user_dict.get("activity_level") or 2)
    diet = generate_diet_recommendation(label, weight, age, activity_level, budget_tier)

    gym_recommendation = None
    try:
        if not df_ref.empty:
            if 'Fitness Goal' in df_ref.columns:
                candidates = df_ref[df_ref['Fitness Goal'].str.lower().str.contains(label.replace("_"," "), na=False)]
                if not candidates.empty:
                    gym_recommendation = candidates.iloc[0].get('Recommendation', None)
            elif 'goal' in df_ref.columns:
                candidates = df_ref[df_ref['goal'].str.contains(label.replace("_"," "), na=False)]
                if not candidates.empty:
                    gym_recommendation = candidates.iloc[0].get('recommendation', None)
    except Exception:
        gym_recommendation = None

    plan = {
        "user_summary": {
            "age": age,
            "sex": sex,
            "height_cm": height,
            "weight_kg": weight,
            "bmi": bmi,
            "sessions_per_week": sessions,
            "budget_tier": budget_tier,
            "health_conditions": health_conditions,
            "existing_activities": existing_activities,
            "predicted_label": label
        },
        "training_setting": choose_training_setting(budget_tier, age, sessions),
        "weekly_workouts": workouts,
        "diet": diet,
        "gym_recommendation_text": gym_recommendation,
        "notes": "Plan generated deterministically from rules + classifier. Tweak sessions_per_week, health_conditions and goal for more precise output."
    }

    return plan
