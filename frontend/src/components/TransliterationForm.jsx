import { useState } from "react";
import { transliterate } from "../api";

export default function TransliterationForm() {
  const [inputWord, setInputWord] = useState("");
  const [outputWord, setOutputWord] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    const result = await transliterate(inputWord);
    setOutputWord(result.output);
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input 
          type="text" 
          value={inputWord} 
          onChange={(e) => setInputWord(e.target.value)} 
          placeholder="Enter Hindi word"
        />
        <button type="submit">Transliterate</button>
      </form>
      {outputWord && <h3>Output: {outputWord}</h3>}
    </div>
  );
}
