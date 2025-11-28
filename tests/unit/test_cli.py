"""Unit tests for containerl.cli with Docker calls mocked out."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import Any

import containerl.cli as cli


def test_is_valid_docker_image_name() -> None:
    assert cli.is_valid_docker_image_name("containerl-test")
    assert cli.is_valid_docker_image_name("registry.example.com/containerl-test:1.0")
    assert not cli.is_valid_docker_image_name("not a valid name")


def test_build_docker_image_nonverbose_success(tmp_path: Any, monkeypatch: Any) -> None:
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
    ) -> Any:
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    image = cli.build_docker_image(
        str(d), name="myimg", tag="v1", verbose=False, context=None
    )
    assert image == "myimg:v1"


def test_build_docker_image_verbose_streams(monkeypatch: Any, tmp_path: Any) -> None:
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


def test_run_docker_container_detached_multiple(monkeypatch: Any) -> None:
    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    def fake_run(
        cmd: Iterable[str],
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
    ) -> Any:
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


def test_run_docker_container_interactive_calls_call(monkeypatch: Any) -> None:
    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    called: dict[str, Any] = {}

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


def test_stop_container_no_containers(monkeypatch: Any) -> None:
    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    def fake_run(
        cmd: Iterable[str],
        capture_output: bool = True,
        text: bool = True,
        check: bool = True,
    ) -> Any:
        # Simulate `docker ps -q --filter ancestor=...` returning empty
        return SimpleNamespace(stdout="\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Should not raise
    cli.stop_container("myimg:latest")


def test_build_run_uses_build_and_run(monkeypatch: Any) -> None:
    def _sleep(_: Any) -> None:  # pragma: no cover
        return None

    monkeypatch.setattr(time, "sleep", _sleep)

    def _build_docker_image(
        path: Any, name: Any, tag: Any, verbose: Any, context: Any
    ) -> str:  # pragma: no cover
        return "img:tag"

    def _run_docker_container(
        *args: Any, **kwargs: Any
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


def test_build_run_test_flow(monkeypatch: Any) -> None:
    # Ensure docker exists
    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    # Patch build_run to avoid building and running
    def _build_run(*a: Any, **k: Any) -> tuple[str, list[str]]:  # pragma: no cover
        return ("img:tag", ["localhost:50051"])

    def _test_connection(*a: Any, **k: Any) -> None:  # pragma: no cover
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
    ) -> Any:
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
        count=1,
    )


def test_main_dispatch_build_and_run(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    # Create tmp Dockerfile
    (tmp_path / "Dockerfile").write_text("FROM alpine:3.12\n")

    def _which(name: str) -> str:  # pragma: no cover
        return "/usr/bin/docker"

    monkeypatch.setattr(shutil, "which", _which)

    called: dict[str, bool] = {}

    def _build_docker_image_cb(
        path: Any, name: Any, tag: Any, verbose: Any, context: Any
    ) -> str:  # pragma: no cover
        called.update({"build": True})
        return "img:tag"

    def _run_docker_container_cb(*a: Any, **k: Any) -> list[str]:  # pragma: no cover
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
