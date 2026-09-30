"""
Sentinel-GL — Common Acquisition Infrastructure and Exceptions.
Governed by Factory v2.2 fail-closed acquisition rules (C54, D-019 - D-023).
"""

from typing import Dict, Any, Optional, List
import socket
import urllib.request
import logging

logger = logging.getLogger("sentinel_gl.acquisition")


class AcquisitionError(Exception):
    """Base exception for data acquisition failures."""
    pass


class AcquisitionBlockedError(AcquisitionError):
    """Raised when external API, network, or authentication failure requires Human action."""
    pass


class AuthenticationError(AcquisitionBlockedError):
    """Raised when provider credentials are missing or invalid."""
    pass


class NetworkReachabilityError(AcquisitionBlockedError):
    """Raised when provider endpoints are unreachable."""
    pass


class RateLimitError(AcquisitionBlockedError):
    """Raised when provider quota is exhausted."""
    pass


def check_endpoint_reachability(host: str, port: int = 443, timeout: float = 5.0) -> bool:
    """Verify network connectivity to external provider endpoint without making data requests."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, socket.gaierror, OSError) as e:
        logger.warning(f"Endpoint {host}:{port} unreachable: {e}")
        return False


def verify_fail_closed_preconditions(service_name: str, host: str) -> None:
    """
    Assert network reachability before attempting acquisition.
    Under NO circumstances may an acquisition script catch this error and substitute manufactured observations.
    """
    if not check_endpoint_reachability(host):
        raise NetworkReachabilityError(
            f"External service '{service_name}' at {host} is unreachable. "
            f"BLOCKED — HUMAN ACTION REQUIRED. Data generation is strictly prohibited per Constitution C54."
        )
