from fastapi import APIRouter
from backend.services.inference import transliterate_word
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Transliteration"])

class TransliterationRequest(BaseModel):
    input_word: str

@router.post("/transliterate")
def transliterate(req: TransliterationRequest):
    output_word = transliterate_word(req.input_word)
    return {"input": req.input_word, "output": output_word}
