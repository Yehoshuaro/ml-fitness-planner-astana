from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from generate_plan_astana import generate_plan

app = FastAPI(title="AI Fitness Planner (local)", version="1.0")

class PlanRequest(BaseModel):
    age: int
    sex: Optional[str] = "Male"
    height_cm: float
    weight_kg: float
    goal: Optional[str] = None
    sessions_per_week: Optional[int] = 3
    budget_kzt: Optional[str] = "0-5000"
    location: Optional[str] = None
    health_conditions: Optional[str] = None
    existing_activities: Optional[List[str]] = None
    activity_level: Optional[int] = 2  # 1..6

@app.post("/plan")
def plan_endpoint(req: PlanRequest):
    payload = req.dict()
    plan = generate_plan(payload)
    return plan

@app.post("/predict")
def predict(req: PlanRequest):
    plan = generate_plan(req.dict())
    return {"recommended_goal": plan["user_summary"]["predicted_label"], "plan": plan}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
