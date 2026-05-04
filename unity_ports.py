"""Helpers for choosing non-conflicting Unity ML-Agents port ranges."""

import socket


def _port_is_free(port: int) -> bool:
    """Match gRPC's wildcard bind behavior when probing candidate ports."""

    sockets = []
    try:
        for family in (socket.AF_INET6, socket.AF_INET):
            try:
                sock = socket.socket(family, socket.SOCK_STREAM)
            except OSError:
                continue

            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            if family == socket.AF_INET6 and hasattr(socket, "IPPROTO_IPV6"):
                try:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                except OSError:
                    pass

            bind_addr = ("::", port) if family == socket.AF_INET6 else ("0.0.0.0", port)
            try:
                sock.bind(bind_addr)
            except OSError:
                return False

        return True
    finally:
        for sock in sockets:
            sock.close()


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
