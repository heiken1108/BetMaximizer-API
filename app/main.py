from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Bet Maximizer API")

app.include_router(router)