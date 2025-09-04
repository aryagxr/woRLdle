from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
import pandas as pd
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import wandb
from peft import get_peft_model, LoraConfig

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto"
).to("cuda")

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",   # attention projections
            "gate_proj", "up_proj", "down_proj"       # MLP projections
        ]
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("predibase/wordle-grpo", split="train")

words = pd.read_csv("https://raw.githubusercontent.com/arnavgarg1/arnavgarg1/refs/heads/main/five_letter_words.csv")
valid_words = set(words["Word"].str.lower())

# helper functions
def regexsearch(text)->str:
    match = re.search(r"<guess>\s*(\w+)\s*</guess>", text)
    return match.group(1).strip().lower() if match else ""

def extractguess(completions)->list[str]:
    guesses = [regexsearch(completion) for completion in completions]
    return guesses

import re
import ast

def finalguess(completions, secret, **kwargs):
    guesses = extractguess(completions)
    answers = [t.strip().lower() for t in secret] * len(completions)
    rewards = [2.0 if g == t and len(g) == 5 else 0.0 for g, t in zip(guesses, answers)]
    return rewards


def partial_correctness(completions, secret, **kwargs):
    guesses = extractguess(completions)
    answers = [t.strip().lower() for t in secret] * len(completions)
    rewards = [
        sum(gc == sc for gc, sc in zip(g, t)) / 5.0 if len(g) == 5 else 0.0
        for g, t in zip(guesses, answers)
    ]
    return rewards


def feedback_reward(completions, past_guess_history, **kwargs):
    guesses = extractguess(completions)
    rewards = []

    for i, g in enumerate(guesses):
        if not g or g.strip() == "":
            rewards.append(0.0)
            continue

        try:
            history_list = ast.literal_eval(past_guess_history[i]) \
                if i < len(past_guess_history) else []

            bonus = 0.0
            penalty = 0.0

            for word, feedback in history_list:
                parsed = re.findall(r"([A-Z])\((.)\)", feedback)

                for pos, (letter, status) in enumerate(parsed):
                    letter = letter.lower()

                    if status == "x":  # should not be present
                        if letter in g:
                            penalty += 0.1
                        else:
                            bonus += 0.05

                    elif status == "-":  # present, wrong position
                        if letter not in g:
                            penalty += 0.1
                        elif pos < len(g) and g[pos] == letter:
                            penalty += 0.1
                        else:
                            bonus += 0.05

                    elif status == "✓":  # correct position
                        if pos >= len(g) or g[pos] != letter:
                            penalty += 0.2
                        else:
                            bonus += 0.1

            reward = max(0.0, 1.0 - penalty + bonus)
        except Exception:
            reward = 0.0

        rewards.append(reward)

    return rewards


def validword(completions, **kwargs):
    guesses = extractguess(completions)
    rewards = []
    for g in guesses:
        penalty = 0.0
        if len(g) != 5:
            penalty += 0.5
        if g not in valid_words:
            penalty += 0.5
        rewards.append(max(0.0, 1.0 - penalty))
    return rewards


def xmlformat(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<guess>.*?</guess>"
    rewards = []
    for completion in completions:
        rewards.append(1.0 if re.search(pattern, completion, flags=re.DOTALL) else 0.0)
    return rewards



training_args = GRPOConfig(output_dir="Qwen2.5-1.5B-GRPO",
                           num_generations=4,
                           log_completions=True,
                           num_completions_to_print=4,
                           logging_steps=1,
                           disable_tqdm=False,
                           learning_rate=1e-4,
                           temperature=0.8,
                           top_p=0.9,
                           top_k=50,
                           gradient_accumulation_steps=1,
                           per_device_train_batch_size=4,
                           report_to="wandb",
                           run_name="wordle-grpo",
                        #    max_grad_norm=0.1,
                           bf16=True,
                           num_train_epochs=5,
                        #    weight_decay = 0.1,
                           save_steps=19,
                        #    adam_beta1 = 0.9,
                        #    adam_beta2 = 0.99,
                           )

trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            finalguess,
            partial_correctness,
            feedback_reward,
            validword,
            xmlformat
            ],
        args=training_args,
        train_dataset=dataset,
    )
trainer.train()