"""Helpers for choosing non-conflicting Unity ML-Agents port ranges."""

import socket


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
        except OSError:
            return False
    return True


def find_free_base_port(
    preferred_base_port: int,
    ports_needed: int,
    search_limit: int = 256,
) -> int:
    """Return the first base port with a fully free contiguous range."""
    if ports_needed <= 0:
        raise ValueError("ports_needed must be positive")

    for base_port in range(preferred_base_port, preferred_base_port + search_limit):
        if all(_port_is_free(base_port + offset) for offset in range(ports_needed)):
            return base_port

    raise RuntimeError(
        "Could not find a free Unity port range starting at "
        f"{preferred_base_port} within {search_limit} attempts."
    )
