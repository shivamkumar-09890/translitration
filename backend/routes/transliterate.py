from backend.services.inference import transliterate_word
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from backend.services.llm import transliterate_to_devanagari
from backend.services.inference_lstm import predict, src_char2idx, tgt_idx2char, DEVICE, model

router = APIRouter(prefix="/api", tags=["Transliteration"])

class TransliterationRequest(BaseModel):
    input_word: str

class TransliterationResponse(BaseModel):
    input: str
    output: str

# Existing transliteration (non-LLM)
@router.post("/transliterate", response_model=TransliterationResponse)
def transliterate_basic(req: TransliterationRequest):
    output_word = transliterate_word(req.input_word)
    return {"input": req.input_word, "output": output_word}

# LLM-based transliteration
@router.post("/transliterate_llm", response_model=TransliterationResponse)
async def transliterate_llm(req: TransliterationRequest):
    """
    Receives a word in Roman script and returns its Devanagari transliteration using OpenAI LLM.
    """
    try:
        output_word = transliterate_to_devanagari(req.input_word)
        if not output_word:
            raise HTTPException(status_code=500, detail="Failed to transliterate the word.")
        return {"input": req.input_word, "output": output_word}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transliteration: {e}")

@router.post("/transliterate_lstm", response_model=TransliterationResponse)
def transliterate_lstm(req: TransliterationRequest):
    """
    Receives a word in Roman script and returns its Devanagari transliteration using the trained LSTM model.
    """
    try:
        output_word = predict(model, req.input_word, src_char2idx, tgt_idx2char)
        if not output_word:
            raise HTTPException(status_code=500, detail="Failed to transliterate the word using LSTM.")
        return {"input": req.input_word, "output": output_word}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LSTM transliteration: {e}")