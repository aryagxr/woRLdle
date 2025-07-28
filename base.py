from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None)
model = model.to("cpu")

def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = """
Task: You are playing Wordle. Your goal is to guess the hidden 5-letter word in as few guesses as possible.

Rules:
1. You have a total of 6 guesses.
2. After each guess, you receive feedback for each letter:
   - "G" = Correct letter in the correct position (Green).
   - "Y" = Correct letter but in the wrong position (Yellow).
   - "B" = Letter not in the word (Black/Grey).
3. Use all previous guesses and feedback to eliminate impossible words and narrow down candidates.
4. Do not repeat impossible letters or invalid words.
5. **You must strictly follow the output format below.**

Output format:
- All reasoning must be placed **only** inside `<think>...</think>`.
- The final guess must be placed **only** inside `<guess>...</guess>`.
- Do not output anything outside these tags.

Example:
<think>
Short reasoning here.
</think>
<guess>VALIDWORD</guess>

---

History:
1. CRANE → B B Y B G
2. GHOST → B Y B B B

Now, suggest your next best guess.

"""

print(chat(prompt))

