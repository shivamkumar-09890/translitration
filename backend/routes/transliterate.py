from fastapi import APIRouter
from backend.services.inference import transliterate_word

router = APIRouter(prefix="/api", tags=["Transliteration"])

@router.post("/transliterate")
def transliterate(input_word: str):
    output_word = transliterate_word(input_word)
    return {"input": input_word, "output": output_word}