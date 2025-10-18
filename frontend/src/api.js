export async function transliterate(inputWord) {
  const res = await fetch("http://localhost:8000/api/transliterate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input_word: inputWord }),
  });
  return res.json();
}
