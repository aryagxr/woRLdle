#Train a model to play Wordle using GRPO and TRL


from __future__ import annotations
import os
import re
import subprocess
from pathlib import Path
import modal

app: modal.App = modal.App("wordle-grpo-trl")


image: modal.Image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_pip_install(
        "trl[vllm]",
        "git+https://huggingface.co/spaces/openenv/wordle",
        "wandb",
        "git+https://github.com/huggingface/transformers.git@main",
        "datasets",
        "jmespath",
    )
)

with image.imports():
    from datasets import Dataset
    from textarena_env import TextArenaAction
    from textarena_env.server.environment import TextArenaEnvironment
    from trl import GRPOConfig, GRPOTrainer

models_dir = Path("/models")
hf_hub_cache_dir = "/root/.cache/huggingface"
checkpoints_volume: modal.Volume = modal.Volume.from_name(
    "wordle-grpo-checkpoints", create_if_missing=True
)
hf_cache_volume: modal.Volume = modal.Volume.from_name(
    "wordle-grpo-hf-cache", create_if_missing=True
)

train_volumes: dict[str, modal.Volume] = {
    str(models_dir): checkpoints_volume,
    hf_hub_cache_dir: hf_cache_volume,
}



model_name = "Qwen/Qwen3-1.7B"
hub_model_id = "wordle-grpo-Qwen3-1.7B"

SYSTEM_PROMPT = """\
You are an expert Wordle solver with deep knowledge of English vocabulary, \
letter frequency patterns, and optimal guessing strategies.

Follow these rules to play Wordle:

1. The target is a 5-letter English word.
2. You have 6 attempts to guess the correct word.
3. After each guess you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position.
   - YELLOW (Y): Letter is in the word but in the wrong position.
   - GRAY (X): Letter is not in the word at all.
4. All guesses must be valid 5-letter English words.
5. You cannot reuse a word you have already guessed.
6. Use the tool `guess` to submit each guess.
"""


class WordleEnv:
    def __init__(self):
        self._env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
        self.reward = 0.0
        self.done = False
        self.green_count = 0
        self.yellow_count = 0
        self.correct = 0.0
        self.repetition_score = 1.0
        self._guesses: set[str] = set()
        self._total_guesses = 0

    def reset(self, **kwargs):
        obs = self._env.reset()

        self._last_full_feedback = obs.messages[0].content
        self.reward = 0.0
        self.done = False
        self.green_count = 0
        self.yellow_count = 0
        self.correct = 0.0
        self.repetition_score = 1.0
        self._guesses = set()
        self._total_guesses = 0
        return self._last_full_feedback

    def _parse_feedback(self, feedback):
        for line in feedback.splitlines():
            tokens = line.strip().split()
            if len(tokens) == 5 and all(t in ("G", "Y", "X") for t in tokens):
                greens = tokens.count("G")
                yellows = tokens.count("Y")
                self.green_count = max(self.green_count, greens)
                self.yellow_count = max(self.yellow_count, yellows)
                if greens == 5:
                    self.correct = 1.0
                return

    def _track_repetition(self, guess):
        self._total_guesses += 1
        word = re.sub(r"[\[\]]", "", guess).strip().lower()
        self._guesses.add(word)
        self.repetition_score = len(self._guesses) / self._total_guesses

    def guess(self, guess):
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            raise ValueError("Game over.")
        obs = self._env.step(TextArenaAction(message=guess))
        full_feedback = obs.messages[0].content
        feedback = full_feedback[len(self._last_full_feedback):]
        self._last_full_feedback = full_feedback
        if "You attempted an invalid move" in feedback:
            self.done = obs.done
            return feedback
        self._parse_feedback(feedback)
        self._track_repetition(guess)
        self.done = obs.done
        return feedback





def reward_correct(environments, **kwargs):
    """1.0 if the model solved the word, 0.0 otherwise."""
    return [env.correct for env in environments]


def reward_greens(environments, **kwargs):
    """Best fraction of greens seen this episode (0.0–1.0)."""
    return [env.green_count / 5.0 for env in environments]


def reward_yellows(environments, **kwargs):
    """Best fraction of yellows seen this episode (0.0–1.0)."""
    return [env.yellow_count / 5.0 for env in environments]


def reward_repetition(environments, **kwargs):
    """Unique valid guesses / total valid guesses (1.0 if no repeats)."""
    return [env.repetition_score for env in environments]


def start_grpo_trainer(use_vllm=False, vllm_mode=None):
    os.environ.setdefault("WANDB_PROJECT", "wordle-grpo")

    dataset: Dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": SYSTEM_PROMPT}]
                for _ in range(3200)
            ]
        }
    )

    grpo_config: GRPOConfig = GRPOConfig(
        output_dir=str(models_dir / hub_model_id),
        save_steps=10,
        save_total_limit=1,
        num_train_epochs=1,
        learning_rate=1e-6,
        gradient_accumulation_steps=48,
        per_device_train_batch_size=1,
        warmup_steps=10,
        optim="adamw_torch",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        num_generations=2,
        max_completion_length=1024,
        log_completions=True,
        num_completions_to_print=2,
        chat_template_kwargs={"enable_thinking": False},
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_gpu_memory_utilization=0.15,
        vllm_max_model_length=3072,
        report_to="wandb",
        run_name=os.environ.get("WANDB_NAME", "wordle-grpo-modal"),
        logging_steps=1,
        reward_weights=[3.0, 1.0, 0.5, 1.5],
        push_to_hub=True,
        hub_model_id=hub_model_id,
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_correct, reward_greens, reward_yellows, reward_repetition],
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=WordleEnv,
    )

    trainer.train()
    trainer.save_model(str(models_dir / hub_model_id))
    trainer.push_to_hub()
    checkpoints_volume.commit()
    hf_cache_volume.commit()


#basic training
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes=train_volumes,
)
def train():
    start_grpo_trainer()


#server mode
@app.function(
    image=image,
    gpu="A100-80GB:2",
    timeout=60 * 60 * 4,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes=train_volumes,
)
def train_vllm_server_mode():
    env_copy = os.environ.copy()
    env_copy["CUDA_VISIBLE_DEVICES"] = "0"  # vLLM server on GPU 0

    subprocess.Popen(
        ["trl", "vllm-serve", "--model", model_name],
        env=env_copy,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # trainer on GPU 1
    start_grpo_trainer(use_vllm=True, vllm_mode="server")


#colocate mode
#USING THIS!
# modal run modal-run.py::train_vllm_colocate_mode
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes=train_volumes,
)
def train_vllm_colocate_mode():
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    start_grpo_trainer(use_vllm=True, vllm_mode="colocate")




#SERVING
#modal deploy modal-run.py

VLLM_PORT: int = 8000

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "vllm==0.12.0",
        "flashinfer-python==0.5.3",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env({"VLLM_USE_V1": "1"})
)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

serve_volumes: dict[str, modal.Volume] = {
    "/root/.cache/vllm": vllm_cache_vol,
    str(models_dir): checkpoints_volume,
    hf_hub_cache_dir: hf_cache_volume,
}


def get_latest_checkpoint_path():
    run_output_dir = models_dir / hub_model_id
    found: list[tuple[int, Path]] = []
    for base in (run_output_dir, models_dir):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            m = re.match(r"^checkpoint-(\d+)$", child.name)
            if m:
                found.append((int(m.group(1)), child))
    if found:
        _step, path = max(found, key=lambda t: t[0])
        return str(path)
    if run_output_dir.is_dir() and (run_output_dir / "config.json").exists():
        return str(run_output_dir)
    raise FileNotFoundError(
        f"No checkpoint-* or config.json under {run_output_dir} (or top-level {models_dir}). "
        "Train once and commit the checkpoints volume before deploying serve."
    )


@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    scaledown_window=15 * 60,
    timeout=10 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes=serve_volumes,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    checkpoint_path = get_latest_checkpoint_path()

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        checkpoint_path,
        "--tokenizer",
        model_name,
        "--served-model-name",
        "wordle-grpo",
        "--max-model-len",
        "8192",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(cmd)
