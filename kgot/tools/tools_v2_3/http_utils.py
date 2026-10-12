# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Shared HTTP helpers for KGoT v2.3 tools.
#
# Patterns follow what proper end-to-end agents do: transient-only retries,
# per-host courtesy delays, secret redaction in user-visible strings,
# SSRF-safe URL validation, and LLM-sized truncation.

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import threading
import time
import urllib.parse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger("Controller.HttpUtils")

KGOT_USER_AGENT = "KGoT/1.1 (https://github.com/spcl/knowledge-graph-of-thoughts)"

TRANSIENT_STATUSES: tuple[int, ...] = (429, 500, 502, 503, 504)

# Per-host courtesy delays (seconds). arXiv's 3s rule is in its API Terms of Use.
_HOST_DELAYS: dict[str, float] = {
    "export.arxiv.org": 3.0,
    "api.semanticscholar.org": 1.1,
    "eutils.ncbi.nlm.nih.gov": 0.4,
}
_DEFAULT_DELAY = 0.15

_throttle_lock = threading.Lock()
_last_call: dict[str, float] = {}

_API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_-]{16,}")
_QUERY_KEY_PATTERN = re.compile(r"([?&](?:key|api_key|api-key)=)[^&\s\"']+")
_BEARER_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/=]{8,}")


def build_retry_session(
    total: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple[int, ...] = TRANSIENT_STATUSES,
) -> requests.Session:
    """Build a requests session that retries transient failures only.

    4xx responses (except 429) are terminal and surface immediately.
    """
    retry = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods={"GET", "POST"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": KGOT_USER_AGENT})
    return session


def _host_of(url_or_host: str) -> str:
    if "://" in url_or_host:
        return (urllib.parse.urlparse(url_or_host).hostname or "").lower()
    return url_or_host.split("/")[0].split(":")[0].lower()


def throttle(url_or_host: str) -> None:
    """Sleep so consecutive calls to one host respect its courtesy delay."""
    host = _host_of(url_or_host)
    delay = _HOST_DELAYS.get(host, _DEFAULT_DELAY)
    with _throttle_lock:
        elapsed = time.monotonic() - _last_call.get(host, 0.0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        _last_call[host] = time.monotonic()


def redact_secrets(text: str) -> str:
    """Redact API keys and bearer tokens from user-visible strings."""
    if not text:
        return text
    text = _API_KEY_PATTERN.sub("AIza[REDACTED]", text)
    text = _QUERY_KEY_PATTERN.sub(r"\1[REDACTED]", text)
    text = _BEARER_PATTERN.sub(r"\1[REDACTED]", text)
    return text


def truncate(text: str | None, limit: int) -> str:
    """Cap text length, marking what was cut."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def strip_html_to_text(html: str | None) -> str:
    """Convert an HTML fragment to collapsed plain text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text(separator=" ").split())


def is_public_url(url: str) -> bool:
    """Check a URL is safe to fetch: http(s), no credentials, public IP only."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.username or parsed.password:
        return False
    host = parsed.hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True
