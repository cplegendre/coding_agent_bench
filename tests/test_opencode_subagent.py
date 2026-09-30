"""OpenCode reviewer configuration across local, queue, and resume paths."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from fastapi import HTTPException

from coding_agent_bench.agents.configs import OpenCodeAgentConfig
from coding_agent_bench.agents.opencode import OpenCodeSubagentConfig
from coding_agent_bench.api import CreateJobRequest, build_cli_command
from coding_agent_bench import api
from coding_agent_bench.builder import HarborCommandBuilder
from coding_agent_bench.cli import app
from coding_agent_bench.job import OpenshiftJob
from coding_agent_bench.resume import _update_agent_endpoint


def configure(subagent=None, **kwargs):
    return OpenCodeAgentConfig().configure(
        model_name="primary",
        server_url="http://primary:8000/v1",
        model_max_len=10000,
        opencode_subagent=subagent,
        **kwargs,
    )


def config(result):
    return json.loads(result.agent_env["OPENCODE_CONFIG_CONTENT"])


def request(**kwargs):
    return CreateJobRequest(
        job_name="reviewer-test",
        agent=kwargs.pop("agent", "opencode"),
        dataset="example/dataset",
        model_name="primary",
        server_url="https://primary.example.com/v1",
        **kwargs,
    )


def test_existing_experiments_unchanged():
    result = configure()
    cfg = config(result)
    assert result.model == cfg["model"] == "vllm/primary"
    assert cfg["provider"]["vllm"]["models"]["primary"]["limit"] == {
        "context": 7500,
        "output": 2500,
    }
    assert "agent" not in cfg
    assert set(cfg["provider"]) == {"vllm"}
    assert set(result.agent_env) == {"OPENCODE_CONFIG_CONTENT"}
    assert request().opencode_subagent is None
    assert "--opencode-subagent" not in build_cli_command(request())


@pytest.mark.parametrize("reviewer_model", ["stronger", "primary"])
def test_independent_models_and_limits(reviewer_model):
    cfg = config(
        configure(
            {
                "model_name": reviewer_model,
                "server_url": "https://reviewer.example.com/",
                "model_max_len": 4000,
                "description": "Ask when stuck",
                "prompt": "Review carefully",
            }
        )
    )
    assert cfg["model"] == "vllm/primary"
    assert cfg["provider"]["vllm"]["options"]["baseURL"] == "http://primary:8000/v1"
    provider = cfg["provider"]["reviewer"]
    assert provider["options"] == {"baseURL": "https://reviewer.example.com/v1"}
    assert provider["models"][reviewer_model]["limit"] == {
        "context": 3000,
        "output": 1000,
    }
    reviewer = cfg["agent"]["reviewer"]
    assert reviewer["mode"] == "subagent"
    assert reviewer["model"] == f"reviewer/{reviewer_model}"
    assert reviewer["description"] == "Ask when stuck"
    assert reviewer["prompt"] == "Review carefully"
    assert reviewer["permission"] == {"edit": "deny", "bash": "deny", "task": "deny"}
    assert cfg["agent"]["build"]["permission"]["task"]["reviewer"] == "allow"


@pytest.mark.parametrize("endpoint", [None, "https://reviewer.example.com"])
def test_resume_updates_only_inherited_reviewer_endpoint(endpoint):
    result = configure({"model_name": "stronger", "server_url": endpoint})
    agent = {"env": result.agent_env}
    _update_agent_endpoint(agent, "http://new-primary:9000")
    cfg = json.loads(agent["env"]["OPENCODE_CONFIG_CONTENT"])
    assert cfg["provider"]["vllm"]["options"]["baseURL"] == "http://new-primary:9000/v1"
    expected = "http://new-primary:9000/v1" if endpoint is None else endpoint + "/v1"
    assert cfg["provider"]["reviewer"]["options"]["baseURL"] == expected


def test_reviewer_openrouter_credentials_and_remote_secret(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-reviewer-key")
    reviewer = {"model_name": "provider/stronger", "server_url": "openrouter"}
    cfg = config(configure(reviewer))
    assert "apiKey" not in cfg["provider"]["vllm"]["options"]
    assert cfg["provider"]["reviewer"]["options"] == {
        "baseURL": "https://openrouter.ai/api/v1",
        "apiKey": "test-reviewer-key",
    }
    command = build_cli_command(request(opencode_subagent=reviewer))
    assert "test-reviewer-key" not in " ".join(command)
    spec = OpenshiftJob("test")._job_spec(command)
    env = spec["spec"]["template"]["spec"]["containers"][0]["env"]
    assert any(item["name"] == "OPENROUTER_API_KEY" for item in env)


def test_missing_reviewer_key_fails(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        configure({"model_name": "stronger", "server_url": "openrouter"})


def test_inherited_openrouter_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = config(
        OpenCodeAgentConfig().configure(
            model_name="primary",
            server_url="openrouter",
            opencode_subagent={"model_name": "stronger"},
        )
    )
    assert cfg["provider"]["reviewer"]["options"] == cfg["provider"]["vllm"]["options"]


@pytest.mark.parametrize("primary_url", ["openrouter", "nebius-h200"])
def test_api_validates_reviewer_even_when_primary_skips_builder(
    primary_url, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    req = request(
        opencode_subagent={"model_name": "stronger", "server_url": "openrouter"}
    )
    req.server_url = primary_url
    with patch.object(HarborCommandBuilder, "build") as build:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.create_job(req))
    assert exc.value.status_code == 400
    assert "OPENROUTER_API_KEY" in exc.value.detail
    build.assert_not_called()


def test_api_validates_explicit_reviewer_url(monkeypatch):
    req = request(
        opencode_subagent={
            "model_name": "stronger",
            "server_url": "https://reviewer.example.com",
        }
    )
    monkeypatch.setattr(
        api, "validate_server_url", lambda url: ["Invalid reviewer endpoint"]
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.create_job(req))
    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid reviewer endpoint"


def test_local_reviewer_does_not_receive_openrouter_secret():
    command = build_cli_command(request(opencode_subagent={"model_name": "stronger"}))
    spec = OpenshiftJob("test")._job_spec(command)
    env = spec["spec"]["template"]["spec"]["containers"][0]["env"]
    assert all(item["name"] != "OPENROUTER_API_KEY" for item in env)


def test_remote_secret_with_equals_cli_option():
    command = [
        "coding-agent-bench",
        "run",
        '--opencode-subagent={"model_name":"stronger","server_url":"openrouter"}',
    ]
    spec = OpenshiftJob("test")._job_spec(command)
    env = spec["spec"]["template"]["spec"]["containers"][0]["env"]
    assert any(item["name"] == "OPENROUTER_API_KEY" for item in env)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"model_name": " "},
        {"model_name": "stronger", "model_max_len": 0},
        {"model_name": "stronger", "model_max_len": "100"},
        {"model_name": "stronger", "server_url": "nebius-h200"},
        {"model_name": "stronger", "server_url": ""},
        {"model_name": "stronger", "description": ""},
        {"model_name": "stronger", "unexpected": True},
    ],
)
def test_invalid_reviewer_config_rejected(value):
    with pytest.raises(ValidationError):
        OpenCodeSubagentConfig.model_validate(value)


@pytest.mark.parametrize("agent", ["oracle", "codex", "pi", "claude-code", "openclaw"])
def test_other_harnesses_reject_option(agent):
    with pytest.raises(ValueError, match="only supported for OpenCode"):
        request(agent=agent, opencode_subagent={"model_name": "stronger"})
    with pytest.raises(ValueError, match="only supported for OpenCode"):
        HarborCommandBuilder().build(
            agent=agent,
            dataset="example/dataset",
            model_name="primary",
            server_url="http://primary:8000",
            environment="docker",
            opencode_subagent={"model_name": "stronger"},
        )


def test_queue_cli_builder_round_trip():
    req = request(opencode_subagent={"model_name": "stronger"})
    command = build_cli_command(req)
    with patch.object(
        HarborCommandBuilder,
        "build",
        return_value=(["harbor", "run"], Path("jobs/test")),
    ) as build:
        result = CliRunner().invoke(app, [*command[1:], "--dry-run"])
    assert result.exit_code == 0, result.output
    reviewer = build.call_args.kwargs["opencode_subagent"]
    assert reviewer == req.opencode_subagent
    harbor, _ = HarborCommandBuilder().build(
        **{key: value for key, value in build.call_args.kwargs.items()}
    )
    env = next(arg for arg in harbor if arg.startswith("OPENCODE_CONFIG_CONTENT="))
    assert (
        json.loads(env.split("=", 1)[1])["agent"]["reviewer"]["model"]
        == "reviewer/stronger"
    )


@pytest.mark.parametrize(
    "agent,reviewer",
    [
        ("opencode", "{broken"),
        ("opencode", "{}"),
        ("oracle", '{"model_name":"stronger"}'),
    ],
)
def test_cli_rejects_invalid_configuration_before_launch(agent, reviewer):
    with patch.object(OpenshiftJob, "run") as launch:
        result = CliRunner().invoke(
            app,
            [
                "run",
                "--agent",
                agent,
                "--dataset",
                "example/dataset",
                "--model-name",
                "primary",
                "--server-url",
                "https://primary.example.com",
                "--environment",
                "openshift",
                "--remote",
                "--opencode-subagent",
                reviewer,
            ],
        )
    assert result.exit_code != 0
    launch.assert_not_called()


def test_example_configuration_loads():
    path = Path(__file__).parents[1] / "examples/opencode-subagent.json"
    req = CreateJobRequest.model_validate_json(path.read_text())
    assert req.agent == "opencode"
    assert req.model_name != req.opencode_subagent.model_name
