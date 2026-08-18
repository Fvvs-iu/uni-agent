# DeepEyes

This example trains a multimodal policy to answer visual questions with an
image-cropping tool. It uses Uni-Agent's decoupled `Task`, `Agent`, and `Tool`
interfaces while the Gateway records the model interactions for verl GRPO.

## Architecture

```text
DeepEyesDataset
  -> TaskConfig serialized in tools_kwargs
  -> run_deepeyes_task
  -> DeepEyesTask
  -> ImageZoomReActAgent
  -> ImageZoomInTool
  -> LLM-as-a-Judge reward
  -> Gateway reward_info
  -> verl TransferQueue
```

The dataset converts each image to a standard OpenAI data URL before the
sample crosses the TransferQueue boundary. The Task owns sample metadata and
reward calculation, the Agent owns the model/tool interaction loop, and the
Tool owns crop validation and execution.

## Requirements

- A working Uni-Agent/verl environment with PyTorch, Ray, vLLM,
  `qwen-vl-utils`, Pillow, and the OpenAI Python client.
- A Qwen multimodal policy checkpoint.
- DeepEyes train and validation parquet files.
- A separate OpenAI-compatible Judge service. The Judge should be capable of
  deciding whether a predicted answer is semantically equivalent to the
  reference answer.

The parquet rows must provide a chat prompt and an image. The recipe accepts
either `<image>` placeholders paired with an `images` column or structured
image content parts. It reads the reference answer from
`reward_model.ground_truth` and the question from `extra_info.question` or the
first user message.

## Train

Run commands from the repository root after preparing the accelerator runtime
and Ray environment:

```bash
export MODEL_PATH=/path/to/qwen-multimodal-policy
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/validation.parquet
export LLM_AS_A_JUDGE_BASE=http://judge-host:port/v1
export LLM_AS_A_JUDGE_MODEL=judge-model-name

bash examples/deepeyes/train_deepeyes.sh
```

The script first sends a small semantic request to the Judge. Set
`CHECK_JUDGE=0` only when that preflight is intentionally handled elsewhere.
Use `DRY_RUN=1` to print the fully resolved training command. Additional Hydra
overrides can be appended to the command line.

The defaults form a one-device smoke configuration. A larger run can override
the topology and batching without editing the script, for example:

```bash
NDEVICES_PER_NODE=8 \
GATEWAY_COUNT=8 \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
ROLLOUT_N=4 \
TOTAL_TRAINING_STEPS=50 \
TEST_FREQ=25 \
bash examples/deepeyes/train_deepeyes.sh
```

The rollout defaults (`max_num_seqs=1`, eager execution, chunked prefill and
prefix caching disabled) are conservative settings used for Qwen3.5 on
vLLM-Ascend. They can be overridden after the target backend has been
validated.

## Reward

The reward follows the DeepEyes recipe:

```text
reward = 0.8 * accuracy + 0.2 * format + 1.2 * tool
```

- `accuracy` is the binary Judge result.
- `format` is `0` for valid output and `-1` for invalid output.
- `tool` is `1` only when at least one crop call succeeds and the final answer
  is correct. A serialized or failed tool call receives no tool bonus.

Validation also reports `acc`, `format`, `tool`, `tool_calls`,
`tool_successes`, `tool_errors`, completion status, and token counts through
`reward_extra_info`.

## Implementation map

- `dataset.py`: parquet adapter and per-sample TaskConfig construction.
- `task.py`: task execution and reward calculation.
- `task_agent.py`: multimodal policy/tool loop.
- `task_tool.py`: image crop tool.
- `reward.py`: Judge prompt, answer parsing, and reward composition.
- `task_runner.py`: task registration and strict reward publication.
- `task_config.yaml`: run-wide Task and Agent defaults.
- `train_deepeyes.sh`: verl v1 GRPO entry point.
