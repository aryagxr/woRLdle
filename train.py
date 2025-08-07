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
    attn_implementation="flash_attention_2",
    device_map="auto"
).to("cuda")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
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

# reward functions
def finalguess(completions, secret, **kwargs) -> list[float]:
    guesses = extractguess(completions)
    # answer = secret.strip().lower()
    answers = [t.strip().lower() for t in secret] * len(completions)
    reward = [2.0 if g == t else 0.0 for g,t in zip(guesses, answers)] 
    # reward = [2.0 if g == answer else 0.0 for g in guesses]
    return reward

def feedback_reward(completions, past_guess_history, **kwargs) -> list[float]:
    rewards = []
    guesses = extractguess(completions)
    
    for i, g in enumerate(guesses):
        penalty = 0.0
        
        history_str = past_guess_history[i] if i < len(past_guess_history) else "[]"
        
        try:
            import ast
            history_list = ast.literal_eval(history_str)
            
            for word, feedback in history_list:
                parsed = re.findall(r"([A-Z])\((.)\)", feedback)
                
                for pos, (letter, status) in enumerate(parsed):
                    letter = letter.lower()
                    
                    if status == "x":
                        if letter in g:
                            penalty += 0.1
                    
                    elif status == "-":
                        if letter not in g:
                            penalty += 0.1
                        elif pos < len(g) and g[pos] == letter:
                            penalty += 0.1
                    
                    elif status == "✓":
                        if pos >= len(g) or g[pos] != letter:
                            penalty += 0.2
            
        except Exception as e:
            penalty = 0.0
        
        reward = max(0.0, 1.0 - penalty)
        rewards.append(reward)
    
    return rewards

def validword(completions, **kwargs) -> list[float]:
    guesses = extractguess(completions)
    rewards = []
    for g in guesses:
        penalty = 0.0
        penalty += 0.5 if len(g) != 5 else 0.0
        penalty += 0.5 if g not in valid_words else 0.0
        reward = max(0.0, 1.0-penalty)
        rewards.append(reward)
    return rewards

def xmlformat(completions, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>\s*<guess>.*?</guess>"
    rewards = []
    
    for completion in completions:
        match = re.search(pattern, completion, flags=re.DOTALL)
        reward = 1.0 if match else 0.0
        rewards.append(reward)
    
    return rewards

training_args = GRPOConfig(output_dir="Qwen2.5-1.5B-GRPO",
                           num_generations=8,
                           log_completions=True,
                           num_completions_to_print=4,
                           learning_rate=5e-6,
                           temperature=0.8,
                           top_p=0.9,
                           top_k=50,
                           gradient_accumulation_steps=8,
                           per_device_train_batch_size=1,
                           report_to="wandb",
                           run_name="wordle-grpo",
                           max_grad_norm=0.1,
                           bf16=True,
                           num_train_epochs=5,
                           weight_decay = 0.1,
                           save_steps=19,
                           adam_beta1 = 0.9,
                           adam_beta2 = 0.99,
                           )

trainer = GRPOTrainer(
        model=model,
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