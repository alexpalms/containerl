"""CLI utilities for building, running, and testing ContaineRL Docker images and containers."""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time

from .interface.utils import AllowedInfoValueTypes

DEFAULT_IMAGE_NAME = "containerl-build"


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the CLI.

    Args:
        verbose: If True, set log level to INFO, otherwise INFO (changed to always show INFO)
    """
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def is_valid_docker_image_name(name: str) -> bool:
    """
    Validate if a string is a valid Docker image name.

    Args:
        name: The image name to validate

    Returns
    -------
        bool: True if valid, False otherwise

    Valid format examples:
    - simple name: containerl-test
    - name with tag: containerl-test:latest
    - with registry: registry.example.com/containerl-test:1.0
    """
    # Basic pattern for docker image names
    pattern = r"""
        ^(?:
            (?:[a-zA-Z0-9](?:[a-zA-Z0-9-._])*/?)*  # Registry and repository name
            [a-zA-Z0-9][a-zA-Z0-9-._]*              # Image name
        )
        (?::[a-zA-Z0-9][-a-zA-Z0-9_.]*)?$          # Optional tag
    """
    return bool(re.match(pattern, name, re.VERBOSE))


def build_docker_image(
    path: str,
    name: str | None = None,
    tag: str | None = None,
    verbose: bool = False,
    context: str | None = None,
) -> str:
    """
    Build a Docker image from a Dockerfile in the specified path.

    Args:
        path: Path to the directory containing the Dockerfile or path to a specific Dockerfile
        name: Optional name for the image
        tag: Optional tag for the image
        verbose: Whether to show detailed build logs
        context: Optional path to use as build context (defaults to current directory)

    Returns
    -------
        The ID of the built image
    """
    logger = logging.getLogger(__name__)

    docker_path = shutil.which("docker")
    if docker_path is None:
        logger.error("Docker executable not found in PATH. Please install Docker.")
        sys.exit(1)

    # Check if path is a file (specific Dockerfile) or directory
    if os.path.isfile(path):
        dockerfile = path
    elif os.path.isdir(path):
        # Use the directory as build context and look for Dockerfile
        dockerfile = os.path.join(path, "Dockerfile")
        if not os.path.isfile(dockerfile):
            raise ValueError(f"No Dockerfile found in {path}")
    else:
        raise ValueError(f"Path {path} does not exist or is not accessible")

    # Use fixed default name if not provided (avoids cache bloat from random names)
    if not name:
        name = DEFAULT_IMAGE_NAME

    # Use 'latest' as default tag if not provided
    if not tag:
        tag = "latest"

    # Create full image name with tag
    image_name = f"{name}:{tag}"

    # Use provided context or default to current directory
    build_context = os.path.abspath(context) if context else "."

    cmd = [docker_path, "build", "-f", dockerfile, "-t", image_name, build_context]

    logger.info(f"Building Docker image with command: {' '.join(cmd)}")

    try:
        if verbose:
            # Use Popen to stream output in real-time when verbose is enabled
            process = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Stream the build logs
            if process.stdout is None:
                raise RuntimeError("Failed to capture Docker build output")

            for line in process.stdout:
                logger.info(line)

            process.wait()
            if process.returncode != 0:
                logger.error(
                    f"Error building Docker image. Process exited with code {process.returncode}"
                )
                sys.exit(1)
        else:
            # Use run with minimal output when not verbose
            logger.info("Building Docker image... (use --verbose for detailed logs)")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
            if result.returncode != 0:
                logger.error(f"Error building Docker image: {result.stderr}")
                sys.exit(1)

        logger.info(f"Successfully built image: {image_name}")
        return image_name
    except Exception as e:
        logger.error(f"Error building Docker image: {str(e)}")
        sys.exit(1)


def run_docker_container(
    image: str,
    port_mapping: bool = True,
    interactive_bash: bool = False,
    additional_args: list[str] | None = None,
    attach: bool = False,
    host_port: int = 50051,
    volumes: list[str] | None = None,
    entrypoint_args: list[str] | None = None,
    count: int = 1,
    container_name: str | None = None,
) -> list[str]:
    """
    Run one or more Docker containers with the specified image and return host addresses.

    Returns a list of addresses (e.g. ['localhost:50051', ...]) for the started containers.
    """
    logger = logging.getLogger(__name__)

    docker_path = shutil.which("docker")
    if docker_path is None:
        logger.error("Docker executable not found in PATH. Please install Docker.")
        sys.exit(1)

    if not is_valid_docker_image_name(image):
        raise ValueError(f"Invalid Docker image name: {image}")

    # Validate that volumes, interactive, attach, and container_name are only used when count=1
    if count > 1:
        if volumes:
            logger.error("Volume mounting is only allowed when count=1")
            sys.exit(1)
        if interactive_bash:
            logger.error("Interactive mode is only allowed when count=1")
            sys.exit(1)
        if attach:
            logger.error("Attach mode is only allowed when count=1")
            sys.exit(1)
        if container_name:
            logger.error("Container naming is only allowed when count=1")
            sys.exit(1)

    addresses: list[str] = []

    # Start 'count' containers sequentially, mapping incremental host ports if needed
    for i in range(count):
        cmd = [docker_path, "run", "--rm"]

        # Add container name if provided
        if container_name:
            cmd.extend(["--name", container_name])

        # Add -it for interactive mode
        if interactive_bash:
            cmd.append("-it")
            # Override entrypoint for interactive mode to ensure we get a shell
            cmd.extend(["--entrypoint", "bash"])
        elif attach:
            # For non-interactive but attached mode, we want to attach to STDOUT
            cmd.extend(["-a", "STDOUT"])
        else:
            # For detached mode (default)
            cmd.append("-d")

        # Add port mapping if enabled. If count > 1, increment host port for each.
        effective_host_port = host_port + i if port_mapping and count > 1 else host_port
        if port_mapping:
            cmd.extend(["-p", f"{effective_host_port}:50051"])

        # Add volume mappings if provided
        if volumes:
            for volume in volumes:
                cmd.extend(["-v", volume])

        if additional_args:
            cmd.extend(additional_args)

        cmd.append(image)

        # Add entrypoint arguments if provided
        if entrypoint_args:
            cmd.extend(entrypoint_args)

        try:
            if interactive_bash:
                # For interactive mode, we use subprocess.call to directly connect
                # the terminal to the container
                logger.info(
                    f"Starting interactive bash session in container from image: {image}"
                )
                subprocess.call(cmd)  # noqa: S603
            elif attach:
                # Using subprocess.Popen with stdout/stderr streaming to console
                process = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )

                logger.info(f"Running container from image: {image}")
                logger.info(f"Command: {' '.join(cmd)}")
                logger.info("Container logs:")

                if process.stdout is None:
                    raise RuntimeError("Failed to capture Docker container output")

                # Stream the output
                for line in process.stdout:
                    logger.info(line)

                process.wait()
                if process.returncode != 0:
                    logger.error(f"Container exited with code {process.returncode}")
                    sys.exit(process.returncode)
            else:
                # Detached mode
                logger.info(f"Starting container in detached mode from image: {image}")
                logger.info(f"Command: {' '.join(cmd)}")
                result = subprocess.run(  # noqa: S603
                    cmd, check=False, capture_output=True, text=True
                )
                if result.returncode != 0:
                    logger.error(f"Error starting container: {result.stderr}")
                    logger.error(f"Error: {result.stderr}")
                    sys.exit(1)
                logger.info("Container started successfully")

                # record the address for this instance
                addresses.append(f"localhost:{effective_host_port}")

        except KeyboardInterrupt:
            logger.info("\nStopping container...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error running Docker container: {str(e)}")
            sys.exit(1)

    # Print addresses to console
    if addresses:
        logger.info("\n=== Container Addresses ===")
        for i, addr in enumerate(addresses, 1):
            logger.info(f"Container {i}: {addr}")

    return addresses


def test_connection(
    server_address: str,
    num_steps: int = 5,
    agent_mode: bool = False,
    init_args: list[str] | None = None,
) -> None:
    """
    Test the connection to a ContaineRL environment or agent server by calling the appropriate client's main function.

    Args:
        server_address: The address of the server (e.g., "localhost:50051")
        num_steps: Number of steps to run in the test
        agent_mode: Whether to test an agent connection instead of an environment connection
        init_args: List of init arguments in "key=value" format
    """
    logger = logging.getLogger(__name__)

    # Parse init_args from "key=value" format to dictionary
    parsed_init_args: dict[str, AllowedInfoValueTypes] = {}
    if init_args:
        for arg in init_args:
            if "=" not in arg:
                logger.error(
                    f"Invalid init argument format: '{arg}'. Expected key=value"
                )
                sys.exit(1)
            key, value = arg.split("=", 1)
            # Try to parse the value as int, float, bool, or keep as string
            if value.lower() in ("true", "false"):
                parsed_init_args[key] = value.lower() == "true"
            elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                parsed_init_args[key] = int(value)
            else:
                try:
                    parsed_init_args[key] = float(value)
                except ValueError:
                    parsed_init_args[key] = value

    try:
        if agent_mode:
            # Import the agent client module's main function
            from containerl import agent_check

            logger.info(f"Testing agent connection to {server_address}...")
            if parsed_init_args:
                logger.info(f"Using init arguments: {parsed_init_args}")
            agent_check(server_address, num_steps=num_steps, **parsed_init_args)
        else:
            # Import the environment client module's main function
            from containerl import environment_check

            logger.info(f"Testing environment connection to {server_address}...")
            if parsed_init_args:
                logger.info(f"Using init arguments: {parsed_init_args}")

            environment_check(
                server_address,
                num_steps=num_steps,
                **parsed_init_args,
            )

        logger.info(
            f"Successfully connected to the {'agent' if agent_mode else 'environment'} server at {server_address}"
        )

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        logger.error(
            f"Failed to connect to or interact with the {'agent' if agent_mode else 'environment'} server."
        )
        sys.exit(1)


def stop_container(image: str | None = None, name: str | None = None) -> None:
    """
    Stop and remove Docker containers by image or by container name.

    Args:
        image: The image name of the containers to stop (optional)
        name: The container name to stop (optional)
    """
    logger = logging.getLogger(__name__)

    if not image and not name:
        logger.error("Either image or name must be provided")
        sys.exit(1)

    if image and name:
        logger.error("Cannot specify both image and name")
        sys.exit(1)

    if image and not is_valid_docker_image_name(image):
        raise ValueError(f"Invalid Docker image name: {image}")

    try:
        # Get full path to docker executable
        docker_path = shutil.which("docker")
        if docker_path is None:
            logger.error("Docker executable not found in PATH. Please install Docker.")
            sys.exit(1)

        if image:
            # Get all container IDs running the specified image
            result = subprocess.run(  # noqa: S603
                [docker_path, "ps", "-q", "--filter", f"ancestor={image}"],
                capture_output=True,
                text=True,
                check=True,
            )
            container_ids = result.stdout.strip().split("\n")
            filter_desc = f"image {image}"
        else:
            # Get container ID(s) by name
            result = subprocess.run(  # noqa: S603
                [docker_path, "ps", "-q", "--filter", f"name={name}"],
                capture_output=True,
                text=True,
                check=True,
            )
            container_ids = result.stdout.strip().split("\n")
            filter_desc = f"name {name}"

        # Filter out empty strings
        container_ids = [cid for cid in container_ids if cid]

        if not container_ids:
            logger.error(f"No running containers found for {filter_desc}")
            return

        logger.info(f"Found {len(container_ids)} container(s) with {filter_desc}")
        for container_id in container_ids:
            try:
                logger.info(f"Stopping container {container_id}...")
                stop_result = subprocess.run(  # noqa: S603
                    [docker_path, "stop", container_id],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if stop_result.returncode != 0:
                    logger.warning(
                        f"Stop command returned non-zero: {stop_result.stderr.strip()}"
                    )

                # Try to remove the container. It might already be removed if started with --rm
                logger.info(f"Removing container {container_id}...")
                rm_result = subprocess.run(  # noqa: S603
                    [docker_path, "rm", "-f", container_id],
                    capture_output=True,
                    text=True,
                    check=False,
                )  # noqa: S603

                if rm_result.returncode == 0:
                    logger.info(
                        f"Container {container_id} stopped and removed successfully"
                    )
                elif "No such container" in rm_result.stderr:
                    # Container was likely started with --rm and auto-removed when stopped
                    logger.info(
                        f"Container {container_id} stopped successfully (auto-removed)"
                    )
                else:
                    logger.error(
                        f"Failed to remove container {container_id}: {rm_result.stderr.strip()}"
                    )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to stop/remove container {container_id}: {e}")

    except Exception as e:
        logger.error(f"Error stopping containers: {str(e)}")
        sys.exit(1)


def build_run(
    path: str,
    name: str | None = None,
    tag: str | None = None,
    port_mapping: bool = True,
    verbose: bool = False,
    context: str | None = None,
    host_port: int = 50051,
    volumes: list[str] | None = None,
    entrypoint_args: list[str] | None = None,
    count: int = 1,
    agent_mode: bool = False,
    container_name: str | None = None,
) -> tuple[str, list[str]]:
    """
    Build a Docker image, run it as one or more containers, and return image and addresses.

    Returns (image, [addresses...])
    """
    """
    Build a Docker image, run it as a container, and test the connection.

    Args:
        path: Path to the directory containing the Dockerfile or path to a specific Dockerfile
        name: Optional name for the image
        tag: Optional tag for the image
        port_mapping: Whether to map port 50051 to the host
        verbose: Whether to show detailed logs
        context: Optional path to use as build context (defaults to current directory)
        host_port: The port on the host to map to container's port 50051 (default: 50051)
        volumes: List of volume mappings in format "host_path:container_path"
        entrypoint_args: List of arguments to pass to the container's entrypoint
    """
    logger = logging.getLogger(__name__)

    try:
        # Step 1: Build the Docker image
        logger.info("=== STEP 1: Building Docker Image ===")
        image = build_docker_image(path, name, tag, verbose, context)

        # Step 2: Run the Docker container(s)
        logger.info("\n=== STEP 2: Running Docker Container(s) ===")
        addresses = run_docker_container(
            image,
            port_mapping,
            False,
            additional_args=None,
            attach=False,
            host_port=host_port,
            volumes=volumes,
            entrypoint_args=entrypoint_args,
            count=count,
            container_name=container_name,
        )

        # Give the containers a moment to start up
        logger.info("Waiting for containers to initialize...")
        time.sleep(3)  # Wait 3 seconds for the containers to start

        # Print addresses and environment variable exports
        if addresses:
            logger.info("\n=== Container Addresses ===")
            for i, addr in enumerate(addresses, 1):
                logger.info(f"Container {i}: {addr}")

            # If agent_mode True, use CONTAINERL_AGENT_ADDRESSES, else CONTAINERL_ENV_ADDRESSES
            env_var = (
                "CONTAINERL_AGENT_ADDRESSES"
                if agent_mode
                else "CONTAINERL_ENV_ADDRESSES"
            )
            joined = ",".join(addresses)
            logger.info(
                "\nYou can export the addresses of the started containers with:"
            )
            logger.info(f"export {env_var}='{joined}'")

        return image, addresses

    except Exception as e:
        logger.error(f"Error during build-run-test sequence: {str(e)}")
        sys.exit(1)


def build_run_test(
    path: str,
    name: str | None = None,
    tag: str | None = None,
    port_mapping: bool = True,
    server_address: str = "localhost:50051",
    num_steps: int = 5,
    verbose: bool = False,
    context: str | None = None,
    host_port: int = 50051,
    volumes: list[str] | None = None,
    entrypoint_args: list[str] | None = None,
    agent_mode: bool = False,
    container_name: str | None = None,
    init_args: list[str] | None = None,
) -> None:
    """
    Build a Docker image, run it as a container, and test the connection.

    Args:
        path: Path to the directory containing the Dockerfile or path to a specific Dockerfile
        name: Optional name for the image
        tag: Optional tag for the image
        port_mapping: Whether to map port 50051 to the host
        server_address: The address of the server (e.g., "localhost:50051")
        num_steps: Number of steps to run in the test
        verbose: Whether to show detailed logs
        context: Optional path to use as build context (defaults to current directory)
        host_port: The port on the host to map to container's port 50051 (default: 50051)
        volumes: List of volume mappings in format "host_path:container_path"
        entrypoint_args: List of arguments to pass to the container's entrypoint
        agent_mode: Whether to test an agent connection instead of an environment connection
        count: Number of containers to start
        container_name: Optional name for the container
        init_args: List of init arguments in "key=value" format
    """
    logger = logging.getLogger(__name__)
    try:
        # Get full path to docker executable
        docker_path = shutil.which("docker")
        if docker_path is None:
            logger.error("Docker executable not found in PATH. Please install Docker.")
            sys.exit(1)

        image, addresses = build_run(
            path,
            name,
            tag,
            port_mapping,
            verbose,
            context,
            host_port,
            volumes,
            entrypoint_args,
            count=1,
            agent_mode=agent_mode,
            container_name=container_name,
        )

        # Step 3: Test the connection (test only first address if multiple)
        logger.info("\n=== STEP 3: Testing Connection ===")

        if addresses:
            test_address = addresses[0]
        else:
            test_address = server_address

        # Test the connection
        test_connection(test_address, num_steps, agent_mode, init_args=init_args)

        # Stop the containers after testing
        logger.info("\n=== Cleaning up: Stopping container(s) ===")
        try:
            container_ids = (
                subprocess.run(  # noqa: S603
                    [docker_path, "ps", "-q", "--filter", f"ancestor={image}"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                .stdout.strip()
                .split("\n")
            )

            container_ids = [cid for cid in container_ids if cid]

            for container_id in container_ids:
                subprocess.run([docker_path, "stop", container_id], check=False)  # noqa: S603
                logger.info(f"Container {container_id} stopped successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Warning: Failed to stop container(s): {e}")

    except Exception as e:
        logger.error(f"Error during build-run-test sequence: {str(e)}")
        sys.exit(1)


def main() -> None:
    """Entry point for the ContaineRL CLI."""
    parser = argparse.ArgumentParser(description="ContaineRL Docker CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a Docker image")
    build_parser.add_argument(
        "path",
        help="Path to the directory containing the Dockerfile or path to a specific Dockerfile",
    )
    build_parser.add_argument(
        "-n", "--name", help="Name for the Docker image (random if not provided)"
    )
    build_parser.add_argument(
        "-t", "--tag", help="Tag for the Docker image (defaults to 'latest')"
    )
    build_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed build logs"
    )
    build_parser.add_argument(
        "-c",
        "--context",
        help="Path to use as build context (defaults to current directory)",
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a Docker container")
    run_parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE_NAME,
        help=f"Docker image to run (default: {DEFAULT_IMAGE_NAME})",
    )
    run_parser.add_argument(
        "--no-port-mapping",
        action="store_true",
        help="Disable mapping port 50051 to the host",
    )
    run_parser.add_argument(
        "--host-port",
        type=int,
        default=50051,
        help="Port on the host to map to container's port 50051 (default: 50051)",
    )
    run_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of containers to start from the same image (default: 1)",
    )
    run_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode with bash shell",
    )
    run_parser.add_argument(
        "-a",
        "--attach",
        action="store_true",
        help="Attach to container's STDOUT (default is detached mode)",
    )
    run_parser.add_argument(
        "--volume",
        action="append",
        dest="volumes",
        help="Mount a volume (format: host_path:container_path). Can be used multiple times.",
    )
    run_parser.add_argument(
        "--entrypoint-args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the container's entrypoint",
    )
    run_parser.add_argument(
        "--docker-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to docker run",
    )
    run_parser.add_argument(
        "--name",
        help="Name for the container",
    )
    run_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed logs"
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Test connection to a ContaineRL environment or agent server"
    )
    test_parser.add_argument(
        "--address",
        default="localhost:50051",
        help="Server address (default: localhost:50051)",
    )
    test_parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of steps to run in the test (default: 5)",
    )
    test_parser.add_argument(
        "--agent",
        action="store_true",
        help="Test agent connection instead of environment connection",
    )
    test_parser.add_argument(
        "--init-arg",
        action="append",
        dest="init_args",
        help="Initialization arguments in key=value format. Can be used multiple times. Supports int, float, bool, and string values.",
    )
    test_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed logs"
    )

    # Build-Run command
    br_parser = subparsers.add_parser(
        "build-run", help="Build a Docker image and run it"
    )
    br_parser.add_argument(
        "path",
        help="Path to the directory containing the Dockerfile or path to a specific Dockerfile",
    )
    br_parser.add_argument(
        "-n", "--name", help="Name for the Docker image (random if not provided)"
    )
    br_parser.add_argument(
        "-t", "--tag", help="Tag for the Docker image (defaults to 'latest')"
    )
    br_parser.add_argument(
        "--no-port-mapping",
        action="store_true",
        help="Disable mapping port 50051 to the host",
    )
    br_parser.add_argument(
        "--host-port",
        type=int,
        default=50051,
        help="Port on the host to map to container's port 50051 (default: 50051)",
    )
    br_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of containers to start from the same image (default: 1)",
    )
    br_parser.add_argument(
        "--agent",
        action="store_true",
        help="Mark the started containers as agents (sets CONTAINERL_AGENT_ADDRESSES when printing exports)",
    )
    br_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed logs"
    )
    br_parser.add_argument(
        "-c",
        "--context",
        help="Path to use as build context (defaults to current directory)",
    )
    br_parser.add_argument(
        "--volume",
        action="append",
        dest="volumes",
        help="Mount a volume (format: host_path:container_path). Can be used multiple times.",
    )
    br_parser.add_argument(
        "--entrypoint-args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the container's entrypoint",
    )
    br_parser.add_argument(
        "--container-name",
        help="Name for the container",
    )

    # Build-Run-Test command
    brt_parser = subparsers.add_parser(
        "build-run-test", help="Build a Docker image, run it, and test the connection"
    )
    brt_parser.add_argument(
        "path",
        help="Path to the directory containing the Dockerfile or path to a specific Dockerfile",
    )
    brt_parser.add_argument(
        "-n", "--name", help="Name for the Docker image (random if not provided)"
    )
    brt_parser.add_argument(
        "-t", "--tag", help="Tag for the Docker image (defaults to 'latest')"
    )
    brt_parser.add_argument(
        "--no-port-mapping",
        action="store_true",
        help="Disable mapping port 50051 to the host",
    )
    brt_parser.add_argument(
        "--address",
        default="localhost:50051",
        help="Server address (default: localhost:50051)",
    )
    brt_parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of steps to run in the test (default: 5)",
    )
    brt_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of containers to start from the same image (default: 1)",
    )
    brt_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed logs"
    )
    brt_parser.add_argument(
        "-c",
        "--context",
        help="Path to use as build context (defaults to current directory)",
    )
    brt_parser.add_argument(
        "--host-port",
        type=int,
        default=50051,
        help="Port on the host to map to container's port 50051 (default: 50051)",
    )
    brt_parser.add_argument(
        "--volume",
        action="append",
        dest="volumes",
        help="Mount a volume (format: host_path:container_path). Can be used multiple times.",
    )
    brt_parser.add_argument(
        "--entrypoint-args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the container's entrypoint",
    )
    brt_parser.add_argument(
        "--agent",
        action="store_true",
        help="Test agent connection instead of environment connection",
    )
    brt_parser.add_argument(
        "--init-arg",
        action="append",
        dest="init_args",
        help="Initialization arguments in key=value format. Can be used multiple times. Supports int, float, bool, and string values.",
    )
    brt_parser.add_argument(
        "--container-name",
        help="Name for the container",
    )

    # Stop command
    stop_parser = subparsers.add_parser(
        "stop", help="Stop and remove running containers by image or name"
    )
    stop_parser.add_argument(
        "--image",
        help="Stop all containers running this image",
    )
    stop_parser.add_argument(
        "--name",
        help="Stop container(s) with this name",
    )
    stop_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed logs"
    )

    args = parser.parse_args()

    if args.command == "build":
        setup_logging(verbose=args.verbose)
        build_docker_image(args.path, args.name, args.tag, args.verbose, args.context)
    elif args.command == "run":
        setup_logging(verbose=args.verbose)
        _ = run_docker_container(
            args.image,
            port_mapping=not args.no_port_mapping,
            interactive_bash=args.interactive,
            additional_args=args.docker_args,
            attach=args.attach,
            host_port=args.host_port,
            volumes=args.volumes,
            entrypoint_args=args.entrypoint_args,
            count=args.count,
            container_name=args.name,
        )
        # Addresses are already printed by run_docker_container
    elif args.command == "test":
        setup_logging(verbose=args.verbose)
        test_connection(
            args.address,
            args.steps,
            args.agent,
            init_args=args.init_args if hasattr(args, "init_args") else None,
        )
    elif args.command == "stop":
        setup_logging(verbose=args.verbose)
        stop_container(image=args.image, name=args.name)
    elif args.command == "build-run":
        setup_logging(verbose=args.verbose)
        _, _ = build_run(
            args.path,
            args.name,
            args.tag,
            not args.no_port_mapping,
            args.verbose,
            args.context,
            args.host_port,
            args.volumes,
            args.entrypoint_args,
            count=args.count,
            agent_mode=args.agent if hasattr(args, "agent") else False,
            container_name=args.container_name,
        )
        # Addresses are already printed by build_run
    elif args.command == "build-run-test":
        setup_logging(verbose=args.verbose)
        build_run_test(
            args.path,
            args.name,
            args.tag,
            not args.no_port_mapping,
            args.address,
            args.steps,
            args.verbose,
            args.context,
            args.host_port,
            args.volumes,
            args.entrypoint_args,
            args.agent,
            container_name=args.container_name,
            init_args=args.init_args if hasattr(args, "init_args") else None,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
