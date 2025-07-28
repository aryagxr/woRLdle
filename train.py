from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re

dataset = load_dataset("predibase/wordle-grpo")

print(dataset)

# Example prompt
prompts = [[
    {"role": "system", "content": "You are playing Wordle..."},
    {"role": "user", "content": "Here is some previous feedback:\nGuess 1: CRANE → x x x - x"}
]]

# Example model output (assistant completion)
completions = [[
    {"role": "assistant", "content": "<think>Let's try based on feedback</think>\n<guess>ROBIN</guess>"}
]]

# Ground truth answer
answers = ["ROBIN"]


def extractguess(text)->str:
    match = re.search(r"<guess>\s*(\w+)\s*</guess>", text)
    return match.group(1).strip().lower() if match else ""


# reward functions
def finalguess(prompts, completions, target, **kwargs) -> list[float]:
    response = [c[0]["content"] for c in completions]
    guesses = [extractguess(r) for r in response]
    return [1.0 if g == t.strip().lower() else 0.0 for g,t in zip(guesses, target)]




def validword(guess):
    return reward

def xmlformat():
    return reward



reward = finalguess(prompts, completions, answers)
print("Reward:", reward)


