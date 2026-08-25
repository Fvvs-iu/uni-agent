# DeepEyes Recipe

This recipe trains a Qwen multimodal policy with GRPO to answer visual
questions. The policy can call `ImageZoomInTool` to crop an image before
returning its final answer, and an OpenAI-compatible Judge provides the
semantic accuracy reward.

## Result

The Qwen3.5-4B experiment uses the fixed validation split generated
by the preprocessing command below. Accuracy is the Judge's binary semantic
correctness score.

| Model | Deepeyes Validation accuracy (%) |
| --- | ---: |
| Qwen3.5-4B before training | 50.0 |
| Qwen3.5-4B after DeepEyes training | 79.2 |
| Absolute improvement | +29.2 |

These numbers are results on the local 48-sample split, not a score on the
full DeepEyes benchmark.

## Requirements

- A working Uni-Agent/verl environment with PyTorch, Ray, vLLM,
  `qwen-vl-utils`, Pillow, and the OpenAI Python client.
- A Qwen multimodal policy checkpoint.
- DeepEyes train and validation parquet files.
- An OpenAI-compatible Judge that can decide whether a prediction is
  semantically equivalent to the reference answer.

Run commands from the repository root. The 4B preset expects eight visible
NPUs by default: devices 0-6 for policy training and rollout, and device 7 for
the Judge.

## 1. Prepare the data

Download the official
[`ChenShawn/DeepEyes-Datasets-47k`](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)
visual-toolbox parquet and create the train/validation split:

```bash
python -m uni_agent.tasks.deepeyes.preprocess \
  --local-save-dir /path/to/deepeyes-data
```

This writes `train.parquet`, `val.parquet`, and `manifest.json`. The default
validation size is 48. To use an existing source parquet without downloading
it again:

```bash
python -m uni_agent.tasks.deepeyes.preprocess \
  --local-save-dir /path/to/deepeyes-data \
  --source-file /path/to/data_0.1.2_visual_toolbox_v2.parquet
```

The manifest records the selected source rows and dataset indices so the split
can be audited.

## 2. Train Qwen3.5-4B

`run_4b_7p1_container.sh` is the reproducible eight-NPU preset used for the
result above. Set the model and data paths if they differ from the defaults in
the script, then launch it inside the prepared training environment:

```bash
POLICY_MODEL=/path/to/Qwen3.5-4B \
JUDGE_MODEL=/path/to/Qwen3.5-4B \
TRAIN_FILE=/path/to/deepeyes-data/train.parquet \
VAL_FILE=/path/to/deepeyes-data/val.parquet \
bash examples/deepeyes/run_4b_7p1_container.sh
```

The launcher starts the Judge on device 7, runs training on devices 0-6, and
stops the Judge when training exits. The resolved configuration, training log,
trajectories, and checkpoints are written under the selected run directory.

Use `DEEPEYES_DRY_RUN=1` to resolve and print the command without starting the
Judge or trainer:

```bash
DEEPEYES_DRY_RUN=1 bash examples/deepeyes/run_4b_7p1_container.sh
```

For other device layouts, use `train_deepeyes.sh` directly and provide the
Judge endpoint:

```bash
MODEL_PATH=/path/to/qwen-multimodal-policy \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
LLM_AS_A_JUDGE_BASE=http://judge-host:port/v1 \
LLM_AS_A_JUDGE_MODEL=judge-model-name \
bash examples/deepeyes/train_deepeyes.sh
```

## Reward

The reward is:

```text
reward = 0.8 * accuracy + 0.2 * format + 1.2 * tool
```

`accuracy` is the binary Judge result. `format` is `0` for valid output and
`-1` for invalid output. `tool` is `1` only when at least one crop succeeds and
the final answer is correct; failed or malformed calls receive no tool bonus.
Judge settings and generation limits are defined in the task config.

## Implementation map

- `dataset.py`: parquet adapter and per-sample task configuration.
- `task_config.yaml`: default task, Judge, agent, and crop-tool settings.
- `task_config_4b.yaml`: settings used by the reported 4B experiment.
- `train_deepeyes.sh`: verl v1 colocate-async GRPO entry point.
- `run_4b_7p1_container.sh`: eight-NPU 4B experiment preset.
- `uni_agent/agents/deepeyes/`: multimodal policy loop and crop tool.
- `uni_agent/tasks/deepeyes/`: preprocessing, task lifecycle, and reward.
