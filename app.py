from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.routes import transliterate

app = FastAPI(title="Transliteration API")

# Register API routes
app.include_router(transliterate.router)

# Serve the frontend HTML
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Optional root endpoint (can redirect to index.html if you want)
@app.get("/api_root")
def root():
    return {"message": "Welcome to Transliteration API"}
