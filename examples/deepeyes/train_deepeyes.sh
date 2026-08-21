#!/usr/bin/env bash
# DeepEyes GRPO training with the verl v1 colocate-async trainer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to a Qwen multimodal policy checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the DeepEyes training parquet}"
: "${VAL_FILE:?Set VAL_FILE to the DeepEyes validation parquet}"
: "${LLM_AS_A_JUDGE_BASE:?Set LLM_AS_A_JUDGE_BASE to an OpenAI-compatible /v1 endpoint}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TASK_CONFIG="${TASK_CONFIG:-examples/deepeyes/task_config.yaml}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"

# Runtime topology. The conservative defaults are suitable for a smoke test;
# increase them together for a full run.
DEVICE="${DEVICE:-npu}"
MODEL_ENGINE="${MODEL_ENGINE:-dp}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-9600}"
NNODES="${NNODES:-1}"
NDEVICES_PER_NODE="${NDEVICES_PER_NODE:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-${NDEVICES_PER_NODE}}"
CONCURRENCY="${CONCURRENCY:-12}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"

# Optimization and rollout.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${NDEVICES_PER_NODE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
ROLLOUT_N="${ROLLOUT_N:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-8}"

# Validation, checkpoints, and logs.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
PROJECT_NAME="${PROJECT_NAME:-deepeyes}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepeyes_grpo}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${REPO_ROOT}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

export LLM_AS_A_JUDGE_BASE
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

if [[ "${DRY_RUN:-0}" != "1" && "${CHECK_JUDGE:-1}" == "1" ]]; then
    "${PYTHON_BIN}" -m uni_agent.tasks.deepeyes.reward
fi

TRAIN_ARGS=(
    --config-name=ppo_trainer
    trainer.use_v1=true
    trainer.v1.trainer_mode=colocate_async
    trainer.v1.colocate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}"
    trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD}"
    trainer.v1.sampler.max_off_policy_strategy=wait
    transfer_queue.enable=true
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.prompt_key=prompt
    data.return_raw_chat=true
    data.return_multi_modal_inputs=false
    data.filter_overlong_prompts=false
    data.truncation=error
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.val_batch_size="${VAL_BATCH_SIZE}"
    data.dataloader_num_workers=0
    data.custom_cls.path=pkg://examples.deepeyes.dataset
    data.custom_cls.name=DeepEyesDataset
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=false
    algorithm.kl_ctrl.kl_coef=0.0
    model_engine="${MODEL_ENGINE}"
    actor_rollout_ref.nccl_timeout="${NCCL_TIMEOUT}"
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.model.use_fused_kernels=false
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=false
    +actor_rollout_ref.actor.use_rollout_log_probs=true
    actor_rollout_ref.actor.use_dynamic_bsz=false
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${NDEVICES_PER_NODE}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.use_kl_loss=false
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.use_torch_compile=false
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}"
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.calculate_log_probs=true
    actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}"
    actor_rollout_ref.rollout.free_cache_engine=true
    actor_rollout_ref.rollout.enable_chunked_prefill=false
    actor_rollout_ref.rollout.enable_prefix_caching=false
    +actor_rollout_ref.rollout.engine_kwargs.vllm.no-enable-chunked-prefill=true
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=false
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.multi_turn.enable=true
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    ++actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}"
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}"
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}"
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=false
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=false
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=true
    reward.reward_manager.name=naive
    trainer.device="${DEVICE}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NDEVICES_PER_NODE}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.logger="['console']"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.resume_mode="${RESUME_MODE}"
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}"
    trainer.default_local_dir="${CHECKPOINT_DIR}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${PYTHON_BIN}" -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}" "$@"
