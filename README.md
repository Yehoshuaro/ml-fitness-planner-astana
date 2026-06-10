# AI Fitness Planner Astana

**Machine Learning-powered API** that generates personalized workout and nutrition plans for people living in Astana (Kazakhstan).

The system takes into account age, gender, fitness goals, health conditions, budget level, activity preferences, and harsh Astana weather conditions.

## Features

- Personalized training program (3–6 days per week)
- Smart training location recommendation (Gym / Home / Outdoor)
- Consideration of health restrictions and injuries
- Budget-aware recommendations
- Rule-based + Machine Learning logic
- FastAPI backend with Swagger documentation
- Local weather adaptation logic (can be extended with real API)

## Tech Stack

- **Backend**: FastAPI + Python
- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Data**: Pandas, survey data + exercise database
- **Deployment**: Ready for Uvicorn / Docker

## Quick Start

```bash
git clone https://github.com/Yehoshuaro/ai-fitness-planner-astana.git
cd ai-fitness-planner-astana

pip install -r requirements.txt
uvicorn app:app --reload
Open: http://127.0.0.1:8000/docs
API Usage Example
POST /plan
JSON{
  "age": 22,
  "sex": "Male",
  "height_cm": 180,
  "weight_kg": 75,
  "goal": "Build muscle",
  "sessions_per_week": 4,
  "budget_kzt": "5000-15000",
  "health_conditions": "knee pain",
  "activity_level": 3
}
```

## Project Structure

app.py — main FastAPI application

generate_plan_astana.py — core logic + plan generation

data/ — datasets and survey results

model/ — trained Random Forest model

train_model.py — model training script
