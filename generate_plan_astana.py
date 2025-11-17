import joblib
import pandas as pd
import os
import math

# ============================
# Load Pre-trained Classifier
# ============================

CLF_PATH = os.path.join("model", "classifier.joblib")
clf = None
if os.path.exists(CLF_PATH):
    clf = joblib.load(CLF_PATH)

# ============================
# Load Datasets
# ============================

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

# ============================
# Helper Functions
# ============================

def safe_float(x, default=None):
    try:
        return float(x)
    except:
        return default

def sex_to_num(x):
    return 1 if str(x).lower().startswith("m") else 0

def calc_bmi(weight, height_cm):
    try:
        h = float(height_cm) / 100.0
        return round(float(weight) / (h * h), 1)
    except Exception:
        return None

def budget_to_tier(budget_kzt):
    if budget_kzt is None:
        return "unknown"
    s = str(budget_kzt).replace(",", "").replace(" ", "")
    try:
        if "-" in s:
            b = float(s.split("-")[-1])
        else:
            b = float(s)
    except:
        return "unknown"

    if b >= 30000:
        return "expensive"
    if 10000 <= b < 30000:
        return "mid"
    if 1 <= b < 10000:
        return "budget"
    return "unknown"

def choose_training_setting(budget_tier, age, schedule_pref):
    if int(age) < 16:
        return {"where": "home_or_outdoor", "reason": "Under 16 → home/outdoor recommended"}
    if budget_tier == "expensive":
        return {"where": "gym", "reason": "Budget allows gym access"}
    if budget_tier == "mid":
        return {"where": "gym_or_home", "reason": "Mid-range budget → flexible options"}
    if budget_tier == "budget":
        return {"where": "home_or_outdoor", "reason": "Budget-limited → home/outdoor workouts"}
    return {"where": "home_or_gym", "reason": "Unknown budget"}

def choose_duration_by_time(schedule_level):
    if schedule_level <= 3:
        return 40
    return 60

# ============================
# Workout Template Builder
# ============================

def build_workout_template(goal_label, availability_per_week, health_conditions, equipment_list, existing_sports):

    workouts = []
    avail = int(max(1, availability_per_week))
    health = ("" if health_conditions is None else str(health_conditions).lower())
    existing = [e.lower() for e in (existing_sports or [])]

    # Rehabilitation
    if "rehab" in goal_label or "rehabil" in health:
        for i in range(avail):
            workouts.append({
                "day_index": i,
                "focus": "rehabilitation / mobility",
                "session_minutes": 30,
                "exercises": [
                    {"name": "controlled hip hinge (no heavy load)", "sets": "3", "reps": "8-12"},
                    {"name": "isometric core activation", "sets": "3", "reps": "20s"},
                    {"name": "banded glute bridges", "sets": "3", "reps": "12-15"},
                    {"name": "shoulder mobility + scapular drills", "sets": "3", "reps": "8-12"}
                ]
            })
        return workouts

    # Asthma / breathing
    if "asthma" in health or "breath" in health:
        cardio_type = "stationary bike / elliptical / indoor cycling"
    else:
        cardio_type = "running / cycling / brisk walking"

    # Muscle Gain
    if "muscle" in goal_label or "gain" in goal_label:
        if avail <= 2:
            for i in range(avail):
                workouts.append({
                    "day_index": i,
                    "focus": "full_body_strength",
                    "session_minutes": choose_duration_by_time(avail),
                    "exercises": [
                        {"name": "squat variation (bodyweight or goblet)", "sets": "3", "reps": "6-12"},
                        {"name": "deadlift or hip hinge (light)", "sets": "3", "reps": "6-10"},
                        {"name": "horizontal push (push-up / bench)", "sets": "3", "reps": "6-12"},
                        {"name": "pull (row / lat pull)", "sets": "3", "reps": "6-12"},
                        {"name": "core stability", "sets": "3", "reps": "20-30s"}
                    ]
                })
        elif avail == 3:
            workouts = [
                {"day_index": 0, "focus": "upper_push_pull", "session_minutes": 50,
                 "exercises": [{"name": "bench / press", "sets": "3", "reps": "6-12"},
                               {"name": "rows", "sets": "3", "reps": "6-12"},
                               {"name": "accessory arms", "sets": "3", "reps": "8-15"}]},
                {"day_index": 1, "focus": "lower_strength", "session_minutes": 50,
                 "exercises": [{"name": "squat / variation", "sets": "4", "reps": "6-10"},
                               {"name": "deadlift / hinge", "sets": "3", "reps": "5-8"},
                               {"name": "calves / core", "sets": "3", "reps": "12-20"}]},
                {"day_index": 2, "focus": "full_body_hypertrophy", "session_minutes": 45,
                 "exercises": [{"name": "compound circuits", "sets": "3", "reps": "8-15"},
                               {"name": "isolation", "sets": "3", "reps": "10-15"}]}
            ]
        else:
            workouts = [
                {"day_index": 0, "focus": "push", "session_minutes": 60,
                 "exercises": [{"name": "bench / overhead press", "sets": "4", "reps": "6-10"},
                               {"name": "dips / push-ups", "sets": "3", "reps": "8-15"}]},
                {"day_index": 1, "focus": "pull", "session_minutes": 60,
                 "exercises": [{"name": "barbell / rows", "sets": "4", "reps": "6-10"},
                               {"name": "pull-ups / lat pulldown", "sets": "3", "reps": "6-12"}]},
                {"day_index": 2, "focus": "legs", "session_minutes": 60,
                 "exercises": [{"name": "squat", "sets": "4", "reps": "6-10"},
                               {"name": "romanian deadlift", "sets": "3", "reps": "6-10"}]},
                {"day_index": 3, "focus": "conditioning_or_accessory", "session_minutes": 40,
                 "exercises": [{"name": cardio_type, "sets": "1", "reps": "20-30min"},
                               {"name": "mobility", "sets": "1", "reps": "10-15min"}]}
            ]
    # Weight Loss
    elif "weight_loss" in goal_label or "lose" in goal_label:
        for i in range(avail):
            workouts.append({
                "day_index": i,
                "focus": "cardio + light strength",
                "session_minutes": choose_duration_by_time(avail),
                "exercises": [
                    {"name": cardio_type, "sets": "1", "reps": "25-40min"},
                    {"name": "circuit bodyweight strength", "sets": "3", "reps": "12-20"}
                ]
            })
    # Endurance
    elif "endurance" in goal_label:
        for i in range(avail):
            workouts.append({
                "day_index": i,
                "focus": "endurance",
                "session_minutes": 40 if avail <= 3 else 60,
                "exercises": [
                    {"name": "steady-state cardio", "sets": "1", "reps": "30-60min"},
                    {"name": "mobility/core", "sets": "2", "reps": "10-20min"}
                ]
            })
    # General fitness
    else:
        for i in range(avail):
            workouts.append({
                "day_index": i,
                "focus": "general_fitness",
                "session_minutes": choose_duration_by_time(avail),
                "exercises": [
                    {"name": "mixed circuit (push/pull/legs)", "sets": "3", "reps": "8-15"},
                    {"name": "short cardio", "sets": "1", "reps": "15-25min"}
                ]
            })

    # Injury modifications
    if "knee" in health:
        for w in workouts:
            for ex in w["exercises"]:
                if "squat" in ex["name"].lower() or "jump" in ex["name"].lower():
                    ex["note"] = "Avoid deep squats or jumps; replace with hip hinge or low-load leg press."
    if "back" in health or "hernia" in health:
        for w in workouts:
            for ex in w["exercises"]:
                if "deadlift" in ex["name"].lower() or "hinge" in ex["name"].lower():
                    ex["note"] = "Use light weights, focus on form, avoid excessive lower back flexion."

    return workouts

# ============================
# Diet Recommendation
# ============================

def generate_diet_recommendation(goal_label, weight_kg, age, activity_level, budget_tier):

    try:
        kcal_baseline = float(weight_kg) * 24
    except:
        kcal_baseline = 2000

    act_mul = 1.2
    if activity_level >= 4:
        act_mul = 1.5
    kcal_total = int(kcal_baseline * act_mul)

    diet = {"estimated_daily_kcal": kcal_total, "macros": {}}

    if "muscle" in goal_label:
        diet["macros"] = {"protein_g": int(2.0 * float(weight_kg)), "carbs_g": "moderate-to-high", "fat_g": "moderate"}
        diet["advice"] = "Increase calories by +200–500 kcal/day, ~2g protein/kg, split into 3–5 meals."
    elif "weight_loss" in goal_label:
        diet["macros"] = {"protein_g": int(1.6 * float(weight_kg)), "carbs_g": "reduced", "fat_g": "moderate"}
        diet["advice"] = "Create a 300–500 kcal/day deficit, maintain 1.6–2g protein/kg, add cardio."
    else:
        diet["macros"] = {"protein_g": int(1.4 * float(weight_kg)), "carbs_g": "balanced", "fat_g": "balanced"}
        diet["advice"] = "Maintenance mode, 3–4 meals/day, ~1.4g protein/kg."

    # Budget-based adjustments
    if budget_tier == "budget":
        diet["budget_advice"] = "Use eggs, cottage cheese, canned fish, and legumes — cheap protein sources."
    elif budget_tier == "mid":
        diet["budget_advice"] = "Add poultry, seasonal vegetables, and grains."
    else:
        diet["budget_advice"] = "You can include more variety and supplements if desired."

    return diet

# ============================
# Main Plan Generator
# ============================

def generate_plan(user_dict: dict):

    age = int(user_dict.get("age", 25))
    sex = user_dict.get("sex", "Male")
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

    # Classifier prediction (if exists)
    predicted_label = None
    if clf is not None:
        try:
            feat = pd.DataFrame([{
                "sex_male": sex_to_num(sex),
                "age": age,
                "height_cm": height,
                "weight_kg": weight,
                "bmi": bmi or -1
            }])
            predicted_label = clf.predict(feat)[0]
        except Exception:
            predicted_label = None

    # Determine final label
    if goal_user_text:
        if "muscl" in goal_user_text or "gain" in goal_user_text:
            label = "muscle_gain"
        elif "lose" in goal_user_text or "weight loss" in goal_user_text or "fat" in goal_user_text:
            label = "weight_loss"
        elif "endurance" in goal_user_text:
            label = "endurance"
        elif "rehab" in goal_user_text or "recover" in goal_user_text:
            label = "rehab"
        else:
            label = predicted_label or "general"
    else:
        label = predicted_label or "general"

    workouts = build_workout_template(label, sessions, health_conditions, None, existing_activities)

    activity_level = int(user_dict.get("activity_level") or 2)
    diet = generate_diet_recommendation(label, weight, age, activity_level, budget_tier)

    gym_recommendation = None
    try:
        if not df_ref.empty:
            if "Fitness Goal" in df_ref.columns:
                candidates = df_ref[df_ref["Fitness Goal"].str.lower().str.contains(label.replace("_", " "), na=False)]
                if not candidates.empty:
                    gym_recommendation = candidates.iloc[0].get("Recommendation", None)
            elif "goal" in df_ref.columns:
                candidates = df_ref[df_ref["goal"].str.contains(label.replace("_", " "), na=False)]
                if not candidates.empty:
                    gym_recommendation = candidates.iloc[0].get("recommendation", None)
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
        "notes": "Plan generated using deterministic logic + optional classifier. Adjust inputs for more precision."
    }

    return plan
