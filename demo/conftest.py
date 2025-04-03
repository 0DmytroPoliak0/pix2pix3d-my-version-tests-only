import sys
sys.path.append('.')

import os
import logging
import pytest
from datetime import datetime

# Configure logging to include timestamps and log levels.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    # Command-line option to indicate CI mode.
    parser.addoption("--ci", action="store_true", default=False, help="Run tests in CI mode.")
    # Additional options (e.g., for boundary or stress tests) can be added here.

@pytest.fixture(scope="session")
def test_config(request):
    """
    Session-scoped fixture that provides a common test configuration.
    This configuration mimics settings relevant to our pix2pix3D project,
    including default image size, seed, model variants, and resource limits.
    """
    config = {
        "image_size": (512, 512),
        "default_seed": 42,
        "models": ["seg2cat", "seg2face", "edge2car"],
        "resource_limits": {
            "max_video_time": 60,   # seconds
            "max_mesh_time": 45,    # seconds
        },
        "ci_mode": request.config.getoption("--ci")
    }
    logger.info("Test configuration: %s", config)
    return config

def pytest_sessionstart(session):
    """
    Hook executed at the very start of the test session.
    Here we can initialize resources, log the start time, or set up global configurations.
    """
    session.start_time = datetime.now()
    logger.info("Test session started at %s", session.start_time)
    # For example, we could load default models or configurations here if needed.
    # This helps simulate our test planning and initialization phase.

def pytest_sessionfinish(session, exitstatus):
    """
    Hook executed after the entire test suite has run.
    Generates a summary report of the test session, including total duration and exit status.
    """
    end_time = datetime.now()
    duration = end_time - session.start_time
    logger.info("Test session finished at %s", end_time)
    logger.info("Total test duration: %s", duration)
    # Write a summary report to a file for later analysis.
    report_file = "test_summary_report.txt"
    with open(report_file, "w") as f:
        f.write(f"Test session started at: {session.start_time}\n")
        f.write(f"Test session finished at: {end_time}\n")
        f.write(f"Total test duration: {duration}\n")
        f.write(f"Exit status: {exitstatus}\n")
    logger.info("Test summary report written to %s", os.path.abspath(report_file))