"""
backend/utils/process.py

Utility for safe subprocess execution.
"""

import subprocess
import logging

logger = logging.getLogger(__name__)


class ProcessError(Exception):
    """Exception raised when a subprocess fails."""
    pass


def safe_subprocess_run(
    cmd,
    timeout=None,
    capture_output=True,
    text=True,
    check=True,
    **kwargs
):
    """
    Safely run a subprocess command.

    Args:
        cmd: Command to run (list of strings)
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
        text: Whether to use text mode
        check: Whether to check return code
        **kwargs: Additional arguments to subprocess.run

    Returns:
        CompletedProcess instance

    Raises:
        ProcessError: If the subprocess fails
    """
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            **kwargs
        )

        if check and result.returncode != 0:
            raise ProcessError(
                f"Command failed with return code {result.returncode}: {result.stderr}"
            )

        return result
    except subprocess.TimeoutExpired as e:
        raise ProcessError(f"Command timed out after {timeout} seconds") from e
    except Exception as e:
        raise ProcessError(f"Subprocess error: {e}") from e
