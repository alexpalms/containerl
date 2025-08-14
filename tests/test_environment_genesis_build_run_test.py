import pytest

from containerl.cli import build_run, stop_container
from containerl.interface.environment.client import main as validate_environment_connection

# Define environments and their build configurations
TEST_CASES = [
    pytest.param("./genesis/drone_hovering/environment/", id="genesis/drone_hovering"),
    pytest.param("./genesis/locomotion/environment/", id="genesis/locomotion"),
]

@pytest.mark.parametrize("env_folder", TEST_CASES)
def test_build_run_environment(env_folder):
    # Run the build_run_test command
    image = build_run(env_folder)

    success = False
    try:
        validate_environment_connection("localhost:50051")
        success = True
    except Exception as e:
        print(f"Error testing environment connection: {str(e)}")

    stop_container(image)

    assert success, "Environment connection failed"
