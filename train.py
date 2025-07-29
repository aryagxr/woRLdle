from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
import pandas as pd

dataset = load_dataset("predibase/wordle-grpo", split="train")
words = pd.read_csv("https://raw.githubusercontent.com/arnavgarg1/arnavgarg1/refs/heads/main/five_letter_words.csv")

#print(dataset)

"""
# Example prompt
prompts = [[
    {"role": "system", "content": "You are playing Wordle..."},
    {"role": "user", "content": "Here is some previous feedback:\nGuess 1: CRANE → x x x - x"}
]]


# Example model output (assistant completion)
completions = [
    [
        {"role": "assistant", "content": "<think>Let's try based on feedback</think>\n<guess>ROBIN</guess>"}
    ],
    [
        {"role": "assistant", "content": "<think>The letter 'I' seems useful here</think>\n<guess>CLEAN</guess>"}
    ],
    [
        {"role": "assistant", "content": "<think>Guessing a word with known letters in new positions</think>\n<guess>BRAIN</guess>"}
    ],
    [
        {"role": "assistant", "content": "<think>Trying a different vowel structure</think>\n<guess>QUILT</guess>"}
    ]
]



# Ground truth answer
answers = ["CLEAN"]
history = [
    ("BLAST", "B(x) L(✓) A(-) S(x) T(x)"),
    ("CLAMP", "C(✓) L(✓) A(-) M(x) P(x)")
]
"""

# helper functions
def regexsearch(text)->str:
    match = re.search(r"<guess>\s*(\w+)\s*</guess>", text)
    return match.group(1).strip().lower() if match else ""


def extractguess(completions)->str:
    response = [c[0]["content"] for c in completions]
    guesses = [regexsearch(r) for r in response]
    return guesses


# reward functions
def finalguess(prompts, completions, target, **kwargs) -> list[float]:
    guesses = extractguess(completions)
    answers = [t.strip().lower() for t in target] * len(completions)
    reward = [2.0 if g == t else 0.0 for g,t in zip(guesses, answers)]
    return reward


def feedback_reward(completions, target, history, **kwargs) -> list[float]:
    rewards = []
    penalty = 0.0
    guesses = extractguess(completions)
    print("Guesses: ", guesses)
    for g in guesses:
        for p,f in history: # past word, feedback
            parsed = re.findall(r"([A-Z])\((.)\)", f)
            for i, (l,s) in enumerate(parsed):
                if s == "x":
                    penalty += 0.1 if l in g else 0.0
                elif s == "-":
                    penalty += 0.1 if l not in g or g[i] == l else 0.0
                elif s == "✓":
                    penalty += 0.2 if g[i]!=l else 0.0

        reward = max(0.0, 1.0-penalty)
        rewards.append(reward)
    return rewards

                    

# check if penalty works, maybe bug
def validword(completions, **kwargs) -> list[float]:
    guesses = extractguess(completions)
    rewards = []
    for g in guesses:
        penalty = 0.0
        penalty += 0.5 if len(g) != 5 else 0.0
        penalty += 0.5 if guess not in words else 0.0
        reward = max(0.0, 1.0-penalty)
        rewards.append(reward)
    return rewards


# check if strict pattern matching needed
# need to check how the completions format output
def xmlformat(completions, **kwargs) -> list[float]:
    pattern = r"^<think>.*?</think>\s*<guess>.*?</guess>$"
    contents = [c[0]["content"] for c in completions]
    return [1.0 if re.match(pattern, text.strip(), flags=re.DOTALL) else 0.0 for text in contents]


reward_funcs = [finalguess, feedback_reward, validword, xmlformat]


training_args = GRPOConfig(output_dir="Qwen2.5-1.5B-GRPO")
trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        reward_funcs=[
            finalguess,
            feedback_reward,
            validword,
            xmlformat
            ],
        args=training_args,
        train_dataset=dataset,
    )
trainer.train()
