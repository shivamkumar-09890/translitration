import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transliterate_to_devanagari(word: str) -> str:
    """
    Takes a word in Roman script and returns its Devanagari transliteration using OpenAI API >=1.0.0.
    """
    prompt = f"Transliterate this word from Roman script to Devanagari: {word}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who transliterates Roman Hindi to Devanagari. just retunr the answer with no statement."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        transliteration = response.choices[0].message.content.strip()
        return transliteration

    except Exception as e:
        print(f"Error during transliteration: {e}")
        return None


# # Example usage
# if __name__ == "__main__":
#     roman_word = "namaste"
#     devnagari_word = transliterate_to_devanagari(roman_word)
#     print(f"{roman_word} -> {devnagari_word}")
