from fastapi import FastAPI
from backend.routes import transliterate

app = FastAPI(title="Transliteration API")

# Register routes
app.include_router(transliterate.router)

@app.get("/")
def root():
    return {"message": "Welcome to Transliteration API"}