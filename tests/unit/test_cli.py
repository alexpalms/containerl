"""Unit tests for containerl.cli with Docker calls mocked out."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import containerl.cli as cli

pytestmark = pytest.mark.unit


def test_is_valid_docker_image_name() -> None:
    assert cli.is_valid_docker_image_name("containerl-test")
    assert cli.is_valid_docker_image_name("registry.example.com/containerl-test:1.0")
    assert not cli.is_valid_docker_image_name("not a valid name")


def test_build_docker_image_nonverbose_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create a temporary directory with a Dockerfile
    d = tmp_path
    df = d / "Dockerfile"
    df.write_text("FROM alpine:3.12\n")

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    def fake_run(
        cmd: Iterable[str],
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    image = cli.build_docker_image(
        str(d), name="myimg", tag="v1", verbose=False, context=None
    )
    assert image == "myimg:v1"


def test_build_docker_image_verbose_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Create Dockerfile
    d = tmp_path
    (d / "Dockerfile").write_text("FROM alpine:3.12\n")

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    class FakeProc:
        def __init__(self) -> None:
            self.returncode = 0
            self._lines = ["Step 1\n", "Step 2\n"]
            self.stdout: Iterator[str] = iter(self._lines)

        def wait(self) -> None:
            return None

    def fake_popen(
        cmd: Iterable[str],
        stdout: int,
        stderr: int,
        universal_newlines: bool,
        bufsize: int,
    ) -> FakeProc:
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    img = cli.build_docker_image(str(d), name="vimg", tag="t", verbose=True)
    assert img == "vimg:t"


def test_run_docker_container_detached_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    def fake_run(
        cmd: Iterable[str],
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    addresses = cli.run_docker_container(
        "myimg:latest",
        port_mapping=True,
        interactive_bash=False,
        attach=False,
        host_port=50051,
        volumes=None,
        entrypoint_args=None,
        count=3,
    )
    assert addresses == ["localhost:50051", "localhost:50052", "localhost:50053"]


def test_run_docker_container_interactive_calls_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    called: dict[str, Iterable[str]] = {}

    def fake_call(cmd: Iterable[str]) -> int:
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    # interactive_bash should use subprocess.call and not raise
    res = cli.run_docker_container(
        "myimg:latest", interactive_bash=True, port_mapping=False, entrypoint_args=None
    )
    # interactive returns empty addresses list since it doesn't append
    assert res == []
    assert "cmd" in called


def test_stop_container_no_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    def fake_run(
        cmd: Iterable[str],
        capture_output: bool = True,
        text: bool = True,
        check: bool = True,
    ) -> SimpleNamespace:
        # Simulate `docker ps -q --filter ancestor=...` returning empty
        return SimpleNamespace(stdout="\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Should not raise
    cli.stop_container(image="myimg:latest")


def test_build_run_uses_build_and_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def _sleep(_: float) -> None:  # pragma: no cover
        return None

    monkeypatch.setattr(time, "sleep", _sleep)

    def _build_docker_image(
        path: Path,
        name: str | None,
        tag: str | None,
        verbose: bool,
        context: str | None,
    ) -> str:  # pragma: no cover
        return "img:tag"

    def _run_docker_container(
        *args: Any, **kwargs: dict[str, Any]
    ) -> list[str]:  # pragma: no cover
        return ["localhost:50051"]

    monkeypatch.setattr(cli, "build_docker_image", _build_docker_image)
    monkeypatch.setattr(cli, "run_docker_container", _run_docker_container)

    image, addrs = cli.build_run(
        "./somepath",
        name=None,
        tag=None,
        port_mapping=True,
        verbose=False,
        context=None,
        host_port=50051,
        volumes=None,
        entrypoint_args=None,
        count=1,
        agent_mode=False,
    )
    assert image == "img:tag"
    assert addrs == ["localhost:50051"]


def test_build_run_test_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure docker exists
    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    # Patch build_run to avoid building and running
    def _build_run(
        *args: Any, **kwargs: dict[str, Any]
    ) -> tuple[str, list[str]]:  # pragma: no cover
        return ("img:tag", ["localhost:50051"])

    def _test_connection(
        *args: Any, **kwargs: dict[str, Any]
    ) -> None:  # pragma: no cover
        return None

    monkeypatch.setattr(cli, "build_run", _build_run)
    # Patch test_connection to be a no-op
    monkeypatch.setattr(cli, "test_connection", _test_connection)

    # Patch subprocess.run used for cleanup to return no containers
    def fake_run(
        cmd: Iterable[str],
        capture_output: bool = True,
        text: bool = True,
        check: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(stdout="\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Should not raise
    cli.build_run_test(
        "./somepath",
        name=None,
        tag=None,
        port_mapping=True,
        server_address="localhost:50051",
        num_steps=1,
        verbose=False,
        context=None,
        host_port=50051,
        volumes=None,
        entrypoint_args=None,
        agent_mode=False,
    )


def test_main_dispatch_build_and_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Create tmp Dockerfile
    (tmp_path / "Dockerfile").write_text("FROM alpine:3.12\n")

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    called: dict[str, bool] = {}

    def _build_docker_image_cb(
        path: Path,
        name: str | None,
        tag: str | None,
        verbose: bool,
        context: str | None,
    ) -> str:  # pragma: no cover
        called.update({"build": True})
        return "img:tag"

    def _run_docker_container_cb(
        *args: Any, **kwargs: dict[str, Any]
    ) -> list[str]:  # pragma: no cover
        called.update({"run": True})
        return ["localhost:50051"]

    monkeypatch.setattr(cli, "build_docker_image", _build_docker_image_cb)
    monkeypatch.setattr(cli, "run_docker_container", _run_docker_container_cb)

    # Test build command
    monkeypatch.setattr(sys, "argv", ["cli", "build", str(tmp_path)])
    cli.main()
    assert called.get("build") is True

    # Test run command
    called.clear()
    monkeypatch.setattr(
        sys, "argv", ["cli", "run", "myimg:latest", "--host-port", "50051"]
    )
    cli.main()
    assert called.get("run") is True


def test_default_image_name_constant() -> None:
    """Test that DEFAULT_IMAGE_NAME constant is defined and used."""
    assert cli.DEFAULT_IMAGE_NAME == "containerl-build"


def test_run_docker_container_rejects_volumes_when_count_gt_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that volumes are rejected when count > 1."""

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    with pytest.raises(SystemExit):
        cli.run_docker_container(
            "myimg:latest",
            volumes=["/host:/container"],
            count=2,
        )


def test_run_docker_container_rejects_interactive_when_count_gt_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that interactive mode is rejected when count > 1."""

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    with pytest.raises(SystemExit):
        cli.run_docker_container(
            "myimg:latest",
            interactive_bash=True,
            count=2,
        )


def test_run_docker_container_rejects_attach_when_count_gt_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that attach mode is rejected when count > 1."""

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    with pytest.raises(SystemExit):
        cli.run_docker_container(
            "myimg:latest",
            attach=True,
            count=2,
        )


def test_stop_container_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test stopping containers by name instead of image."""

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    calls: list[list[str]] = []

    def fake_run(
        cmd: Iterable[str],
        capture_output: bool = True,
        text: bool = True,
        check: bool = True,
    ) -> SimpleNamespace:
        cmd_list = list(cmd)
        calls.append(cmd_list)
        # Simulate finding a container
        if "ps" in cmd_list:
            return SimpleNamespace(stdout="abc123\n")
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    cli.stop_container(name="my-container")

    # Check that the filter used name instead of ancestor
    assert any("name=my-container" in str(call) for call in calls)


def test_stop_container_requires_image_or_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stop_container requires either image or name."""

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    with pytest.raises(SystemExit):
        cli.stop_container()


def test_stop_container_rejects_both_image_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that stop_container rejects both image and name."""

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    with pytest.raises(SystemExit):
        cli.stop_container(image="myimg:latest", name="my-container")


def test_run_docker_container_with_container_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that container_name adds --name flag to docker command."""

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    calls: list[list[str]] = []

    def fake_run(
        cmd: Iterable[str],
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
    ) -> SimpleNamespace:
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cli.run_docker_container(
        "myimg:latest",
        container_name="my-container",
        port_mapping=False,
        count=1,
    )

    # Check that --name flag was added to the command
    assert len(calls) == 1
    cmd = calls[0]
    assert "--name" in cmd
    name_idx = cmd.index("--name")
    assert cmd[name_idx + 1] == "my-container"


def test_run_docker_container_rejects_container_name_when_count_gt_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that container_name is rejected when count > 1."""

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    with pytest.raises(SystemExit):
        cli.run_docker_container(
            "myimg:latest",
            container_name="my-container",
            count=2,
        )
