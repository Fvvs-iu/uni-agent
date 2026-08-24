from __future__ import annotations

from pathlib import Path

from PIL import Image

from examples.deepeyes.dataset import DeepEyesDataset
from uni_agent.agents.deepeyes import DeepEyesAgentConfig
from uni_agent.tasks import TaskConfigResolver, get_task
from uni_agent.tasks.deepeyes import DeepEyesTask


def _dataset_for(row: dict) -> DeepEyesDataset:
    dataset = DeepEyesDataset.__new__(DeepEyesDataset)
    dataset.dataframe = [row]
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    return dataset


def test_dataset_emits_serializable_task_config():
    dataset = _dataset_for(
        {
            "data_source": "deepeyes/test",
            "prompt": [{"role": "user", "content": "<image>What animal is shown?"}],
            "images": [Image.new("RGB", (20, 20), "white")],
            "reward_model": {"ground_truth": "cat"},
            "extra_info": {"question": "What animal is shown?", "index": 7},
        }
    )

    row = dataset[0]

    assert [message["role"] for message in row["raw_prompt"]] == ["user"]
    image_part = row["raw_prompt"][0]["content"][0]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    task_config = row["tools_kwargs"]["task"]
    assert task_config["name"] == "deepeyes"
    assert task_config["question"] == "What animal is shown?"
    assert task_config["ground_truth"] == "cat"
    assert task_config["data_source"] == "deepeyes/test"
    assert task_config["metadata"] == {"index": 7}
    assert "prompt" not in task_config
    assert "images" not in row
    assert "agent_name" not in row


def test_dataset_structured_image_placeholder_consumes_images_column():
    dataset = _dataset_for(
        {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What animal is shown?"},
                    ],
                }
            ],
            "images": [Image.new("RGB", (32, 32), "white")],
            "reward_model": {"ground_truth": "cat"},
            "extra_info": {"question": "What animal is shown?"},
        }
    )

    row = dataset[0]

    assert row["raw_prompt"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_recipe_config_resolves_core_deepeyes_task_and_agent():
    root = Path(__file__).resolve().parents[3]
    resolver = TaskConfigResolver.from_file(str(root / "examples/deepeyes/task_config.yaml"))
    resolved = resolver.resolve(
        {
            "name": "deepeyes",
            "prompt": [{"role": "user", "content": "question"}],
            "question": "question",
            "ground_truth": "answer",
        },
        runtime_model={
            "base_url": "http://gateway/v1",
            "api_key": "EMPTY",
            "model_name": "policy",
        },
    )

    task = get_task(resolved)

    assert isinstance(task, DeepEyesTask)
    assert isinstance(task.config.agent, DeepEyesAgentConfig)
    assert task.config.agent.name == "deepeyes"
    assert task.config.agent.model.base_url == "http://gateway/v1"


def test_training_script_uses_generic_task_runner():
    root = Path(__file__).resolve().parents[3]
    script = (root / "examples/deepeyes/train_deepeyes.sh").read_text()

    assert "exec \"${PYTHON_BIN}\" -m verl.trainer.main_ppo" in script
    assert "job submit --no-wait" not in script
    assert "runner_fqn=uni_agent.framework.task_runner.run_task" in script
    assert "runner_kwargs.report_reward=true" in script
    assert "examples.deepeyes.task_runner" not in script
