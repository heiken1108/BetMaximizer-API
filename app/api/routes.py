from fastapi import APIRouter, HTTPException
from app.services.services import PredictionService

router = APIRouter()
predictions_service = PredictionService()

@router.get("/")
async def read_root():
	return {"health": "OK"}

#/predict?home_team=<Home Team>&away_team=<Away Team>
@router.get("/predict")
async def predict(home_team: str, away_team: str):
	home_team = home_team.replace("_", " ")
	away_team = away_team.replace("_", " ")
	pred = predictions_service.perform_prediction(home_team, away_team)
	return {"home_team": home_team, "away_team": away_team, "prediction": pred}

