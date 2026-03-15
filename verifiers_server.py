"""
Verifiers Environment wrapper for Atropos.

Enables Verifiers environments to be used for Atropos Evaluation, Data Generation and Training (SFT/RL).
Uses verifiers' native rollout for correct multi-turn support and token capture.
"""

import json
import logging
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

# Verifiers imports
import verifiers as vf
from openai import AsyncOpenAI
from pydantic import Field, Json
from verifiers.types import RolloutInput, RolloutOutput
from verifiers.utils.message_utils import concat_messages

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.envs.server_handling.managed_server import ManagedServerAdapter

# Ensure logs are visible
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifiers_bridge")


def _to_json_safe(obj: Any) -> Any:
    """Convert object to JSON-serializable form."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return str(obj)


class VerifiersEnvConfig(BaseEnvConfig):
    """Configuration for VerifiersEnv."""

    vf_env_name: str = Field(
        ..., description="Verifiers environment name (e.g., 'gsm8k', 'wordle')"
    )
    env_args: Union[Dict[str, Any], Json[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Arguments passed to vf.load_environment()",
    )
    extra_env_kwargs: Union[Dict[str, Any], Json[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Runtime arguments passed to vf_env.set_kwargs()",
    )
    # Atropos config overrides
    max_token_length: int = Field(default=2048, description="Max generation tokens")
    group_size: int = Field(
        default=4, description="Number of samples to generate per prompt"
    )
    # Eval specific
    num_examples: int = Field(
        default=5, description="Number of examples to evaluate (eval mode only)"
    )
    rollouts_per_example: int = Field(
        default=1, description="Rollouts per example (eval mode only)"
    )
    eval_temperature: Optional[float] = Field(
        default=None,
        description="Temperature for generation (None = use server default)",
    )


class VerifiersEnv(BaseEnv):
    """
    Atropos environment that wraps any Verifiers Environment Hub environment.

    Uses verifiers' native rollout for correct multi-turn support.
    Supports both Evaluation (verifiers-driven) and Training/DataGen (atropos-driven).
    """

    name = "verifiers"
    env_config_cls = VerifiersEnvConfig

    def __init__(
        self,
        config: VerifiersEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: VerifiersEnvConfig = config

        # Load Verifiers Env
        self.vf_env = vf.load_environment(config.vf_env_name, **config.env_args)
        if config.extra_env_kwargs:
            self.vf_env.set_kwargs(**config.extra_env_kwargs)

        # Dataset iteration state
        self.dataset = None
        self.iter = 0

        # AsyncOpenAI client for verifiers rollout (lazy init in setup)
        self._vf_client: Optional[AsyncOpenAI] = None

        # Buffers for logging metrics to WandB
        self.reward_buffer = []
        self.metrics_buffer = defaultdict(list)

    @classmethod
    def config_init(cls) -> Tuple[VerifiersEnvConfig, List[APIServerConfig]]:
        """Default configuration for CLI usage."""
        env_config = VerifiersEnvConfig(
            vf_env_name="gsm8k",
            env_args={},
            tokenizer_name="Qwen/Qwen2.5-1.5B",
            max_token_length=2048,
            group_size=4,
            batch_size=64,
            wandb_name="verifiers-bridge",
            use_wandb=True,
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B",
                base_url="http://localhost:8000/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """Initialize dataset iterator and AsyncOpenAI client for verifiers."""
        if self.dataset is None:
            self.dataset = self.vf_env.get_dataset().shuffle(
                seed=random.randint(0, 10000)
            )
            self.iter = 0
            logger.info(f"Loaded dataset with {len(self.dataset)} examples")

        # Create AsyncOpenAI client for verifiers rollout
        if self._vf_client is None and self.server and self.server.servers:
            server_config = self.server.servers[0].config
            limits = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            http_client = httpx.AsyncClient(
                limits=limits, timeout=server_config.timeout
            )
            self._vf_client = AsyncOpenAI(
                base_url=server_config.base_url,
                api_key=server_config.api_key or "x",
                http_client=http_client,
            )
            logger.info(
                f"Created AsyncOpenAI client for verifiers: {server_config.base_url}"
            )

    async def get_next_item(self) -> Item:
        """Get next item for Atropos data generation loop."""
        if self.dataset is None:
            await self.setup()

        item = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1
        return item

    def _item_to_rollout_input(self, item: Item, example_id: int = 0) -> RolloutInput:
        """Convert Atropos Item to verifiers RolloutInput."""
        return RolloutInput(
            prompt=item.get("prompt", []),
            example_id=example_id,
            task=item.get("task", "default"),
            answer=item.get("answer", ""),
            info=item.get("info", {}),
        )

    def _build_scored_data(
        self, states: List[RolloutOutput], tracked_nodes: List[Any]
    ) -> Optional[ScoredDataGroup]:
        """
        Merge verifiers rewards with ManagedServer perfectly aligned tokens.
        Matches concurrent rollout states to their respective ManagedServer nodes
        to prevent alignment bugs caused by asynchronous generation completion.
        """
        if not states or not tracked_nodes:
            return None

        scored_data: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "messages": [],
            "inference_logprobs": [],
            "overrides": [],
            "group_overrides": {},
            "advantages": None,
            "ref_logprobs": None,
            "generation_params": None,
            "images": None,
            "distill_token_ids": None,
            "distill_logprobs": None,
        }

        # Keep track of which nodes we haven't matched yet
        unmatched_nodes = list(tracked_nodes)

        for state in states:
            if state.get("error") is not None:
                continue

            completion = state.get("completion", [])
            if not completion:
                continue

            # Safely extract final text content for matching
            if isinstance(completion, str):
                final_content = completion
            elif isinstance(completion, list) and len(completion) > 0:
                final_content = str(completion[-1].get("content", ""))
            else:
                final_content = ""

            # Match the verifiers state to the ManagedServer node by content suffix
            matched_node = None
            for node in unmatched_nodes:
                if node.full_text.endswith(final_content):
                    matched_node = node
                    break

            if matched_node is None:
                logger.warning(
                    "Could not find matching ManagedServer node for verifiers state. Skipping."
                )
                continue

            # Remove it so we don't accidentally match it twice
            unmatched_nodes.remove(matched_node)

            # --- Now we can safely merge them! ---

            reward = state.get("reward", 0.0)

            # Capture metrics for WandB logging
            self.reward_buffer.append(reward)

            # Verifiers states often contain a breakdown of metrics (e.g. strict accuracy vs partial)
            state_metrics = state.get("metrics", {})
            if state_metrics:
                for k, v in state_metrics.items():
                    if isinstance(v, (int, float)):
                        self.metrics_buffer[k].append(v)

            prompt = state.get("prompt") or []
            # Safely concatenate whether they are strings or lists
            full_messages = concat_messages([prompt, completion])

            # Data from ManagedServer (Perfectly aligned tokens/masks/logprobs)
            scored_data["tokens"].append(matched_node.tokens)
            scored_data["masks"].append(matched_node.masked_tokens)
            scored_data["inference_logprobs"].append(matched_node.logprobs)

            # Data from Verifiers (Rewards, metrics, messages)
            scored_data["scores"].append(reward)
            scored_data["messages"].append(full_messages)
            scored_data["overrides"].append({})

        # Return None if all rollouts failed
        if not scored_data["tokens"]:
            return None

        return scored_data

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Generate and score trajectories using verifiers' native rollout.

        Delegates to vf_env.run_group which handles:
        - Multi-turn loops
        - Stop conditions
        - State tracking
        - Scoring
        """
        if self._vf_client is None:
            await self.setup()

        rollout_inputs = [
            self._item_to_rollout_input(item, example_id=i)
            for i in range(self.config.group_size)
        ]

        # Sampling args for generation
        sampling_args = {
            "max_tokens": self.config.max_token_length,
        }
        if self.config.eval_temperature is not None:
            sampling_args["temperature"] = self.config.eval_temperature

        # 1. Atropos automatically finds the least-busy GPU in the cluster
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            # 2. Create the dummy OpenAI client for Verifiers
            adapter_client = ManagedServerAdapter(
                managed, base_url=managed.server.config.base_url
            )

            try:
                # 3. Verifiers runs its multi-turn logic using the adapter
                outputs = await self.vf_env.run_group(
                    group_inputs=rollout_inputs,
                    client=adapter_client,
                    model="default",
                    sampling_args=sampling_args,
                )
            except Exception as e:
                logger.error(f"Error during verifiers rollout: {e}")
                return None, []

            # 4. Atropos has been secretly tracking the perfectly aligned tokens/logprobs!
            tracked_nodes = managed.get_state()["nodes"]

        # 5. Zip them together.
        scored_data = self._build_scored_data(list(outputs), tracked_nodes)

        return scored_data, []

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate using verifiers' built-in evaluation logic.
        """
        start_time = time.time()
        server_config = self.server.servers[0].config

        limits = httpx.Limits(max_connections=28000, max_keepalive_connections=28000)
        http_client = httpx.AsyncClient(limits=limits, timeout=server_config.timeout)

        # Create client for verifiers evaluation
        client = AsyncOpenAI(
            base_url=server_config.base_url,
            api_key=server_config.api_key or "x",
            http_client=http_client,
            max_retries=10,
        )

        sampling_args = {"max_tokens": self.config.max_token_length}
        if self.config.eval_temperature is not None:
            sampling_args["temperature"] = self.config.eval_temperature

        results = await self.vf_env.evaluate(
            client=client,
            model=server_config.model_name,
            sampling_args=sampling_args,
            num_examples=self.config.num_examples,
            rollouts_per_example=self.config.rollouts_per_example,
            max_concurrent=getattr(self.config, "max_eval_workers", 32),
        )

        end_time = time.time()
        avg_reward = results["metadata"]["avg_reward"]
        avg_metrics = results["metadata"]["avg_metrics"]

        samples = []
        for output in results["outputs"]:
            sample = {
                "prompt": output.get("prompt"),
                "completion": output.get("completion"),
                "reward": output.get("reward"),
                "answer": output.get("answer"),
                "stop_condition": output.get("stop_condition"),
            }

            if output.get("error"):
                err = output["error"]
                sample["error"] = (
                    err.get("error_chain_str", str(err))
                    if isinstance(err, dict)
                    else str(err)
                )

            samples.append(sample)

        await self.evaluate_log(
            metrics={
                "eval/avg_reward": avg_reward,
                **{f"eval/{k}": v for k, v in avg_metrics.items()},
            },
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters=sampling_args,
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Override base logger to include Verifiers-specific metrics.
        """
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log average reward
        if self.reward_buffer:
            avg_reward = sum(self.reward_buffer) / len(self.reward_buffer)
            wandb_metrics["metrics/mean_reward"] = avg_reward
            self.reward_buffer = []

        # Log custom metrics from the Verifiers environment (e.g. accuracy, format_score)
        if self.metrics_buffer:
            for metric_name, values in self.metrics_buffer.items():
                if values:
                    avg_metric = sum(values) / len(values)
                    # Prefix with train/ to keep dashboard clean
                    wandb_metrics[f"metrics/{metric_name}"] = avg_metric
            self.metrics_buffer = defaultdict(list)

        # Call parent to handle server metrics, perf stats, and the rollouts table
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    VerifiersEnv.cli()
