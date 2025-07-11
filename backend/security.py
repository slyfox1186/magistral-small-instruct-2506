#!/usr/bin/env python3
"""Enhanced Security Gateway for Web-Connected LLM Agent.

Provides robust URL validation, SSRF protection, and resource limiting
for a locally-hosted LLM with full internet access capabilities.
"""

import ipaddress
import logging
import socket
from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Constants
MAX_REQUEST_LOG_SIZE = 1000


class URLValidator:
    """Enhanced URL validation and SSRF protection for web agent."""

    # Allowed schemes for web scraping
    ALLOWED_SCHEMES: ClassVar[set[str]] = {"http", "https"}

    # Private/internal IP ranges to block (RFC 1918, RFC 3927, etc.)
    BLOCKED_IP_RANGES: ClassVar[list[str]] = [
        "127.0.0.0/8",  # Loopback
        "10.0.0.0/8",  # Private Class A
        "172.16.0.0/12",  # Private Class B
        "192.168.0.0/16",  # Private Class C
        "169.254.0.0/16",  # Link-local (AWS metadata)
        "224.0.0.0/4",  # Multicast
        "240.0.0.0/4",  # Reserved
        "::1/128",  # IPv6 loopback
        "fc00::/7",  # IPv6 unique local
        "fe80::/10",  # IPv6 link-local
    ]

    # Blocked hostnames for additional security
    BLOCKED_HOSTNAMES: ClassVar[set[str]] = {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",  # noqa: S104 - This is blocked for security, not listening
        "::1",
        "metadata.google.internal",  # GCP metadata
        "169.254.169.254",  # AWS/Azure metadata
    }

    def __init__(self):
        """Initialize URLValidator with blocked network ranges."""
        self._blocked_networks = []
        for range_str in self.BLOCKED_IP_RANGES:
            try:
                self._blocked_networks.append(ipaddress.ip_network(range_str, strict=False))
            except ValueError:
                continue

    def is_valid_url(self, url: str) -> bool:
        """Enhanced URL validation with SSRF protection.

        Args:
            url: The URL to validate

        Returns:
            True if the URL is safe to access, False otherwise
        """
        # Early validation checks
        validation_result = self._validate_url_basic_checks(url)
        if validation_result is not None:
            return validation_result

        try:
            parsed = urlparse(url)

            # Check scheme and hostname
            if not self._is_scheme_allowed(parsed.scheme):
                return False

            if not self._is_hostname_valid(parsed.hostname):
                return False

            # Check for blocked hostnames/domains
            hostname = parsed.hostname.lower()
            if hostname in self.BLOCKED_HOSTNAMES:
                logger.warning(f"Blocked hostname: {hostname}")
                return False

            # Check IP addresses
            return self._check_ip_addresses(hostname)

        except Exception as e:
            logger.warning(f"URL validation failed for {url}: {e}")
            return False

    def _validate_url_basic_checks(self, url: str) -> bool | None:
        """Perform basic URL validation checks. Returns None if validation should continue."""
        if not url or not isinstance(url, str):
            logger.warning("Invalid URL: empty or non-string")
            return False
        return None

    def _is_scheme_allowed(self, scheme: str) -> bool:
        """Check if URL scheme is allowed."""
        if scheme.lower() not in self.ALLOWED_SCHEMES:
            logger.warning(f"Blocked scheme: {scheme}")
            return False
        return True

    def _is_hostname_valid(self, hostname: str) -> bool:
        """Check if hostname exists."""
        if not hostname:
            logger.warning("No hostname in URL")
            return False
        return True

    def _check_ip_addresses(self, hostname: str) -> bool:
        """Check if hostname resolves to blocked IP addresses."""
        try:
            # Get all IP addresses for the hostname
            addr_info = socket.getaddrinfo(hostname, None)
            for _family, _type, _proto, _canonname, sockaddr in addr_info:
                ip_str = sockaddr[0]
                try:
                    ip_addr = ipaddress.ip_address(ip_str)
                    for blocked_network in self._blocked_networks:
                        if ip_addr in blocked_network:
                            logger.warning(f"Blocked IP {ip_str} in range {blocked_network}")
                            return False
                except ValueError:
                    # Invalid IP address
                    continue
        except socket.gaierror as e:
            # DNS resolution failed
            logger.warning(f"DNS resolution failed for {hostname}: {e}")
            return False
        else:
            return True

    def sanitize_url_for_display(self, url: str) -> str:
        """Sanitize URL for safe display in logs/UI.

        Args:
            url: The URL to sanitize

        Returns:
            Sanitized URL safe for display
        """
        if not self.is_valid_url(url):
            return "[INVALID URL]"

        try:
            parsed = urlparse(url)
            # Remove any sensitive query parameters
            safe_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        except Exception:
            return "[UNPARSEABLE URL]"
        else:
            return safe_url

    def sanitize_url_for_shell(self, url: str) -> str:
        """Sanitize URL for safe use in shell commands.

        Args:
            url: The URL to sanitize

        Returns:
            URL safe for shell usage
        """
        # Ensure it's a string
        if not isinstance(url, str):
            url = str(url)

        # Basic shell escaping - wrap in single quotes and escape any single quotes
        escaped = url.replace("'", "'\"'\"'")
        return f"'{escaped}'"


class WebRequestLimiter:
    """Resource limiting for web requests."""

    def __init__(self):
        """Initialize WebRequestLimiter with default limits."""
        self.max_response_size = 10 * 1024 * 1024  # 10MB
        self.max_redirects = 5
        self.request_timeout = 30  # seconds
        self.allowed_content_types = {
            "text/html",
            "text/plain",
            "application/json",
            "application/xml",
            "text/xml",
        }

    def is_content_type_allowed(self, content_type: str) -> bool:
        """Check if content type is allowed."""
        if not content_type:
            return False

        # Extract main content type (ignore charset, etc.)
        main_type = content_type.split(";")[0].strip().lower()
        return main_type in self.allowed_content_types


class SecurityGateway:
    """Main security gateway for web agent operations."""

    def __init__(self):
        """Initialize SecurityGateway with validation and limiting components."""
        self.url_validator = URLValidator()
        self.request_limiter = WebRequestLimiter()
        self.request_log: list[dict[str, Any]] = []

    def validate_web_request(self, url: str, user_approved: bool = False) -> dict[str, Any]:
        """Validate a web request before execution.

        Args:
            url: URL to validate
            user_approved: Whether user has approved this request

        Returns:
            Dict with validation result
        """
        result = {
            "approved": False,
            "url": url,
            "sanitized_url": self.url_validator.sanitize_url_for_display(url),
            "timestamp": datetime.now(UTC).isoformat(),
            "reasons": [],
        }

        # Basic URL validation
        if not self.url_validator.is_valid_url(url):
            result["reasons"].append("URL failed security validation")
            return result

        # User approval check (if required)
        if not user_approved:
            result["reasons"].append("User approval required for web access")
            return result

        result["approved"] = True
        return result

    def log_request(self, url: str, status: str, details: str = ""):
        """Log web request for audit trail."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "url": self.url_validator.sanitize_url_for_display(url),
            "status": status,
            "details": details,
        }
        self.request_log.append(log_entry)

        # Keep only last entries as defined by constant
        if len(self.request_log) > MAX_REQUEST_LOG_SIZE:
            self.request_log = self.request_log[-MAX_REQUEST_LOG_SIZE:]

        logger.info(f"Web request {status}: {log_entry['url']} - {details}")


# Global instances for easy importing
url_validator = URLValidator()
security_gateway = SecurityGateway()
