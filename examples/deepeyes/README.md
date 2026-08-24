# DeepEyes

This example trains a multimodal policy to answer visual questions with an
image-cropping tool. The reusable DeepEyes `Task`, `Agent`, and `Tool` live in
`uni_agent`; this directory contains only the verl training recipe.

## Architecture

```text
DeepEyesDataset
  -> TaskConfig serialized in tools_kwargs
  -> uni_agent.framework.task_runner.run_task
  -> DeepEyesTask
  -> DeepEyesAgent
  -> ImageZoomInTool
  -> LLM-as-a-Judge reward
  -> Gateway reward_info
  -> verl TransferQueue
```

The dataset emits an Agent-neutral user message and converts each image to a
standard OpenAI data URL before the sample crosses the TransferQueue boundary.
The recipe owns the system prompt and Agent/Tool settings, the Task owns sample
metadata and reward calculation, the Agent owns the model/tool interaction
loop, and the Tool owns crop validation and execution.

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

## Prepare data

Download the official visual-toolbox Parquet and create a training/validation
split:

```bash
python -m uni_agent.tasks.deepeyes.preprocess \
  --local-save-dir /path/to/deepeyes-data
```

This writes `train.parquet`, `val.parquet`, and `manifest.json`. The manifest
records the selected source row positions and dataset indices. To prepare an
already downloaded source file without network access, use:

```bash
python -m uni_agent.tasks.deepeyes.preprocess \
  --local-save-dir /path/to/deepeyes-data \
  --source-file /path/to/data_0.1.2_visual_toolbox_v2.parquet
```

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

The script first sends a small semantic request to the Judge and then runs the
verl trainer in the foreground. Set `CHECK_JUDGE=0` only when that preflight is
intentionally handled elsewhere. Use `DRY_RUN=1` to print the fully resolved
training command. Additional Hydra overrides can be appended to the command
line.

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

The formula weights are fixed in `reward.py`. The answer-length limit, Judge
timeout/retry behavior, strict mode, and generation parameters are declared
under `reward` in `task_config.yaml`. Judge `base_url`, `model_name`, and
`api_key` can also be declared there; when omitted, they are resolved from
`LLM_AS_A_JUDGE_BASE`, `LLM_AS_A_JUDGE_MODEL`, and
`LLM_AS_A_JUDGE_API_KEY`.

The generic task runner reports reward, accuracy, and completion status to the
training framework. The Task also retains format, tool-use, and token metrics
in `TaskResult.extra_info` and the runtime logs.

## Implementation map

- `dataset.py`: parquet adapter and per-sample TaskConfig construction.
- `task_config.yaml`: run-wide Task and Agent defaults.
- `train_deepeyes.sh`: foreground verl v1 GRPO entry point.
- `uni_agent/agents/deepeyes/agent.py`: multimodal policy loop and message conversion.
- `uni_agent/agents/deepeyes/tool.py`: image decoding, crop validation, and crop tool.
- `uni_agent/tasks/deepeyes/task.py`: sample lifecycle and reward invocation.
- `uni_agent/tasks/deepeyes/reward.py`: configurable Judge and reward composition.
- `uni_agent/tasks/deepeyes/preprocess.py`: official dataset download and train/validation split.
