"""The command line, run directories and the .env file."""

import json
import os
import subprocess

import pytest

pytest.importorskip("rclpy", reason="needs the ROS environment (source install/setup.bash)")

from sail_pilot import REPO, output  # noqa: E402
from sail_pilot.cli import default_memory, load_dotenv, main  # noqa: E402


def test_cli_writes_a_complete_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(output, "RUNS_DIR", tmp_path)
    code = main(["landgraben", "--target", "orange", "--pilot", "scripted",
                 "--no-video", "--quiet"])
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    summary = json.loads((run_dir / "run.json").read_text())["summary"]
    assert summary["success"] is True
    lines = (run_dir / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == summary["decisions"] == len(list((run_dir / "frames").iterdir()))
    first = json.loads(lines[0])
    assert {"state", "prompt", "turn_deg", "latency", "new_heading"} <= set(first)
    for name in ("trace.csv", "trajectory.png", "recording.npz", "scenario.yaml"):
        assert (run_dir / name).is_file()


@pytest.mark.parametrize("argv, message", [
    (["--model", "claude-opus-5-5", "--no-thinking"], "cannot be turned off"),
    (["--model", "claude-sonnet-5-5", "--no-thinking", "--effort", "max"], "high or below"),
])
def test_impossible_thinking_settings_are_refused(argv, message, capsys):
    with pytest.raises(SystemExit):
        main(["landgraben", "--target", "orange", *argv])
    assert message in capsys.readouterr().err


def test_memory_is_on_for_instructions_only():
    assert default_memory("Round the island.", None) is True
    assert default_memory(None, "green") is False


def test_runs_started_together_get_their_own_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(output, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(output.time, "strftime", lambda fmt: "2026-01-01_00-00-00")
    a, b = output.new_run_dir("x"), output.new_run_dir("x")
    assert a != b and b.name.endswith("_2")
    assert (a / "frames").is_dir() and (b / "frames").is_dir()


# --- .env --------------------------------------------------------------------

def test_dotenv_sets_missing_variables_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nANTHROPIC_API_KEY='sk-test-123'\n"
                   "export OTHER_SETTING=\"two words\"\nALREADY=from-file\nEMPTY=\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_SETTING", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    monkeypatch.setenv("ALREADY", "from-environment")
    loaded = load_dotenv(env)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-123"
    assert os.environ["OTHER_SETTING"] == "two words"
    assert os.environ["ALREADY"] == "from-environment"      # the environment wins
    assert "EMPTY" not in os.environ
    assert sorted(loaded) == ["ANTHROPIC_API_KEY", "OTHER_SETTING"]


def test_missing_dotenv_is_fine(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_dotenv_is_ignored_by_git_and_docker():
    assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO).returncode == 0
    assert subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=REPO).returncode == 1
    assert ".env" in (REPO / ".dockerignore").read_text().split()
