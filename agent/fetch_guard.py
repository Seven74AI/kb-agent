"""SSRF guards for server-side URL fetches (stdlib urllib, curl, Flaresolverr)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def _reject_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Reject addresses that must not be reached from server-side fetches."""
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError(f"fetch URL host is not allowed (resolved to {ip})")


def _assert_fetch_url_allowed(url: str) -> None:
    """Block SSRF targets before urllib opens the connection."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("fetch URL must use http or https")
    host = parsed.hostname
    if not host:
        raise ValueError("fetch URL must include a host")

    bracketed = host.startswith("[") and host.endswith("]")
    host_for_ip = host[1:-1] if bracketed else host
    try:
        ip = ipaddress.ip_address(host_for_ip)
    except ValueError:
        pass
    else:
        _reject_blocked_ip(ip)
        return

    try:
        infos = socket.getaddrinfo(
            host, parsed.port or 0, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as exc:
        raise ValueError(f"fetch URL host could not be resolved: {host}") from exc
    if not infos:
        raise ValueError(f"fetch URL host could not be resolved: {host}")

    seen: set[str] = set()
    for family, _type, _proto, _canon, sockaddr in infos:
        if family == socket.AF_INET:
            ip = ipaddress.ip_address(sockaddr[0])
        elif family == socket.AF_INET6:
            ip = ipaddress.ip_address(sockaddr[0])
        else:
            continue
        key = str(ip)
        if key in seen:
            continue
        seen.add(key)
        _reject_blocked_ip(ip)
