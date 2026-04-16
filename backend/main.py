from fastapi import FastAPI
from medalix.api.routes import router

app = FastAPI(title="MedAIx Backend", version="0.1.0")

app.include_router(router)


@app.get("/")
def root():
    return {"message": "MedAIx backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}