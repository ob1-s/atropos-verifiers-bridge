# Atropos Verifiers Bridge

The environment works as a bridge between Atropos and the [Verifiers](https://github.com/primeintellect-ai/verifiers) library, as well with the [Environment Hub](https://app.primeintellect.ai/dashboard/environments). It is designed to work out-of-the-box with `atropos` trainers.

## Installation

This bridge is designed to be a lightweight plugin for your existing Atropos environment. There are no extra dependencies to install other than having a working `atroposlib` and `verifiers` environment.

1. Ensure you have the Prime CLI tool (`prime`) and the specific environment you wish to run:

```bash
uv tool install prime

# Install a specific environment (e.g., Alphabet Sort)
prime env install primeintellect/alphabet-sort
```

## Serving the Environment

Start the environment worker pointing to your desired Verifiers environment. Many Verifiers environments require specific configuration (like dataset splits, turn limits, etc.), which you can pass directly via the `--env.env_args` parameter.

```bash
python verifiers_server.py serve \
    --env.vf_env_name alphabet-sort \
    --env.group_size 8 \
    --openai.base_url http://localhost:9001/v1 \
    --openai.api_key x \
    --openai.model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --env.use_wandb true \
    --env.wandb_name "alphabet-sort" \
    --env.env_args '{"min_turns": 3, "max_turns": 5, "power_per_turn": false}'
```

## Training with GRPO using the example trainer.

1. Start Rollout Server.

```bash
run-api
```

2. Start trainer.

```bash
# Provide the path to your Atropos training script
python example_trainer/grpo.py
```

3. Ensure the Environment Worker is running (as shown in Step 2).

## SFT Data Generation

If you want to generate high-quality reasoning traces for SFT using a Verifiers environment, use the `atropos-sft-gen` CLI:

```bash
atropos-sft-gen path/to/output.jsonl --tokenizer Qwen/Qwen2.5-1.5B-Instruct
```
