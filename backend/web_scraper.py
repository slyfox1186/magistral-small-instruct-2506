#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import secrets
import sys
import time
from typing import Any

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import utils
from circuit_breaker import CircuitBreakerError, get_web_scraper_breaker
from config import CACHE_CONFIG, EXTERNAL_SERVICES
from security import url_validator
from utils import format_prompt

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# Get Google API credentials from environment variables
GOOGLE_API_KEY = EXTERNAL_SERVICES["google_api_key"]
GOOGLE_CSE_ID = EXTERNAL_SERVICES["google_cse_id"]

# Cache for storing scraped content
_scrape_cache = {}
CACHE_TTL_SECONDS = CACHE_CONFIG["scrape_cache_ttl"]

# Constants for HTTP status codes and limits
HTTP_OK = 200
HTTP_MULTIPLE_CHOICES = 300
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500
CONTENT_LENGTH_LIMIT = 10000
CONTENT_PREVIEW_LIMIT = 3000
MIN_CONTENT_LENGTH = 100
MAX_SEARCH_RESULTS = 10
# Common browser headers
COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",  # Do Not Track
}
# Enhanced headers with more recent browser signature and
# additional headers to reduce 403 errors
ENHANCED_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",  # Pretend we came from Google search
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
    "DNT": "1",
    "Cache-Control": "max-age=0",
    "sec-ch-ua": '"Not A(Brand";v="99", "Microsoft Edge";v="120", "Chromium";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}
# Alternative headers to try if the first set fails (rotate user agents)
ALTERNATIVE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


async def validate_url_accessibility(url: str, timeout_seconds: int = 5) -> bool:
    """Quickly validate if URL is accessible before attempting full scraping.

    Performs a lightweight HEAD request to check:
    - 404 Not Found
    - 403 Forbidden
    - 500+ Server errors
    - Connection timeouts

    Args:
        url: URL to validate
        timeout_seconds: Request timeout

    Returns:
        True if URL appears accessible, False otherwise.
    """
    try:
        timeout_config = aiohttp.ClientTimeout(total=timeout_seconds)
        async with (
            aiohttp.ClientSession(timeout=timeout_config, headers=COMMON_HEADERS) as session,
            session.head(url) as response,
        ):
            status = response.status
            logger.debug(f"URL validation for {url}: HTTP {status}")

            # Consider these status codes as inaccessible
            error_statuses = [
                HTTP_NOT_FOUND,
                HTTP_FORBIDDEN,
                HTTP_INTERNAL_SERVER_ERROR,
                501,
                502,
                503,
                504,
            ]

            if status in error_statuses:
                logger.warning(f"URL {url} returned error status: {status}")
                is_accessible = False
            elif status == HTTP_UNAUTHORIZED:
                # 401 Unauthorized - might still be scrapeable with different headers
                logger.info(f"URL {url} requires authentication (401), will try scraping anyway")
                is_accessible = True
            elif HTTP_OK <= status < HTTP_BAD_REQUEST:
                # 200-299 range is good, 300-399 redirects are usually fine
                is_accessible = True
            else:
                # Any other status code, be conservative and skip
                logger.warning(f"URL {url} returned unexpected status: {status}")
                is_accessible = False

            return is_accessible

    except (TimeoutError, aiohttp.ClientError, Exception) as e:
        if isinstance(e, TimeoutError):
            logger.warning(f"URL validation timeout for {url}")
        elif isinstance(e, aiohttp.ClientError):
            logger.warning(f"URL validation failed for {url}: {e}")
        else:
            logger.warning(f"Unexpected error validating {url}: {e}")
        return False


async def scrape_website(
    url: str,
    timeout_seconds: int = 8,  # Reduced from 15 to 8 seconds
    max_retries: int = 2,  # Reduced from 4 to 2 retries
    base_backoff_delay: float = 1.0,  # Reduced from 2.0 to 1.0
    max_jitter: float = 0.5,  # Reduced jitter
) -> str | None:
    """Asynchronously scrapes a website's textual content using aiohttp with retries and exponential backoff.

    Enhanced with better anti-blocking measures and authentication detection.
    Protected by circuit breaker to prevent cascading failures.
    Will not attempt to bypass authentication requirements and will return an appropriate
    message when authentication is detected.

    Args:
        url: The URL to scrape.
        timeout_seconds: Timeout in seconds for the request.
        max_retries: Maximum number of retries.
        base_backoff_delay: Base delay for exponential backoff.
        max_jitter: Maximum random jitter to add to backoff delay.

    Returns:
        The scraped text content of the website, or None if scraping fails.
        Returns a special message if authentication is required.
    """
    # Get circuit breaker for web scraping
    breaker = get_web_scraper_breaker()

    try:
        return await breaker.call(
            _scrape_website_internal,
            url,
            timeout_seconds,
            max_retries,
            base_backoff_delay,
            max_jitter,
        )
    except CircuitBreakerError as e:
        logger.warning(f"Web scraping circuit breaker is open: {e}")
        return "Web scraping service is temporarily unavailable. Please try again later."
    except Exception as e:
        logger.exception("Web scraping failed")
        return f"Web scraping failed: {e!s}"


def _detect_authentication_required(soup: BeautifulSoup, response: aiohttp.ClientResponse, page_text: str | None = None) -> bool:
    """Detect if authentication is required based on HTML content and response headers.

    Args:
        soup: BeautifulSoup parsed HTML content
        response: HTTP response object
        page_text: Optional pre-extracted page text (for performance)

    Returns:
        True if authentication is required, False otherwise
    """
    # Check for login forms
    login_forms = soup.find_all(
        "form",
        id=lambda x: x and ("login" in x.lower() or "signin" in x.lower()),
    )
    login_forms += soup.find_all(
        "form",
        class_=lambda x: x and ("login" in x.lower() or "signin" in x.lower()),
    )
    login_forms += soup.find_all(
        "form",
        action=lambda x: x
        and (
            "login" in x.lower() or "signin" in x.lower() or "auth" in x.lower()
        ),
    )

    # Check for password fields
    password_fields = soup.find_all("input", {"type": "password"})

    # Check for common login elements
    login_buttons = soup.find_all(
        ["button", "input", "a"],
        string=lambda s: s and ("login" in s.lower() or "sign in" in s.lower()),
    )
    login_buttons += soup.find_all(
        ["button", "input", "a"],
        value=lambda s: s and ("login" in s.lower() or "sign in" in s.lower()),
    )

    # Check for common authentication text in the page
    if page_text is None:
        page_text = soup.get_text().lower()

    auth_phrases = [
        "please log in",
        "please sign in",
        "login required",
        "authentication required",
        "you must be logged in",
        "please authenticate",
        "username and password",
        "create an account",
        "member login",
        "account required",
    ]
    has_auth_text = any(phrase in page_text for phrase in auth_phrases)

    # Check response headers for auth indicators
    auth_headers = response.headers.get("WWW-Authenticate") is not None

    # Determine if authentication is required
    return (
        bool(login_forms)
        or bool(password_fields)
        or bool(login_buttons)
        or has_auth_text
        or auth_headers
        or response.status in {HTTP_UNAUTHORIZED, HTTP_FORBIDDEN}
    )


def _extract_and_clean_content(soup: BeautifulSoup, url: str) -> str:
    """Extract and clean text content from HTML.

    Args:
        soup: BeautifulSoup parsed HTML content
        url: Source URL for logging

    Returns:
        Cleaned text content with length limiting
    """
    # Remove script and style elements for text extraction
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Get the main text content
    content = soup.get_text(separator="\n", strip=True)

    # Limit content length to avoid overwhelming responses
    if len(content) > CONTENT_LENGTH_LIMIT:
        content = (
            content[:CONTENT_LENGTH_LIMIT]
            + "\n\n[Content truncated for brevity...]"
        )

    return content


def _add_cache_busting_params(url: str) -> str:
    """Add timestamp parameter to URL for cache busting.

    Args:
        url: Original URL

    Returns:
        URL with cache busting parameter
    """
    timestamp = int(time.time())
    if "?" not in url:
        return f"{url}?_t={timestamp}"
    elif not url.endswith("&") and not url.endswith("?"):
        return f"{url}&_t={timestamp}"
    else:
        return f"{url}_t={timestamp}"


async def _try_aiohttp_scrape(
    url: str,
    timeout_seconds: int,
    headers: dict[str, str]
) -> tuple[str | None, bool]:
    """Try to scrape using aiohttp. Returns (content, is_auth_required)."""
    url_with_param = _add_cache_busting_params(url)
    async with aiohttp.ClientSession(
        headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_seconds)
    ) as session, session.get(url_with_param, allow_redirects=True) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            if "html" in content_type:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, "html.parser")

                if _detect_authentication_required(soup, response):
                    return None, True

                return _extract_and_clean_content(soup, url), False
            elif "text" in content_type:
                return await response.text(), False
            else:
                logger.warning(f"Unsupported content type '{content_type}' for {url}")
                return None, False


async def _try_curl_fallback(url: str) -> str | None:
    """Try to scrape using curl as fallback."""
    mobile_user_agent = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 "
        "Mobile/15E148 Safari/604.1"
    )
    header_args = [
        "-H", f"User-Agent: {mobile_user_agent}",
        "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "-H", "Accept-Language: en-US,en;q=0.9",
        "-H", "Referer: https://www.bing.com/search",
        "-H", "Cookie: visited=true; session=temp",
        "-H", "sec-ch-ua-mobile: ?1",
        "-H", 'sec-ch-ua-platform: "iOS"',
    ]

    try:
        safe_url = url_validator.sanitize_url_for_shell(url)
    except ValueError as e:
        logger.exception("URL validation failed")
        return f"Error: Invalid or unsafe URL - {e!s}"

    # Add cache busting parameter
    safe_url = _add_cache_busting_params(safe_url)

    process = await asyncio.create_subprocess_exec(
        "curl", "-s", "-L", "-v", *header_args, safe_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    logger.debug(f"Curl debug info: {stderr.decode('utf-8', errors='ignore')[:500]}")

    if process.returncode == 0 and stdout:
        html_content = stdout.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")

        if _detect_authentication_required(soup, None):
            return None

        # Clean content
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text(separator="\n", strip=True)

    return None


def _cache_and_return(url: str, content: str | None) -> str | None:
    """Cache content and return it."""
    _scrape_cache[url] = {"timestamp": time.time(), "content": content}
    return content


async def _handle_403_error(url: str, retries: int, max_retries: int) -> str | None:
    """Handle 403 Forbidden errors with curl fallback."""
    if retries < max_retries - 1:
        return None  # Continue retrying

    logger.info(f"Trying curl fallback for {url} after 403 Forbidden")

    try:
        curl_content = await _try_curl_fallback(url)
        if curl_content:
            logger.info(f"Successfully scraped with curl fallback for {url}")
            return curl_content
    except Exception:
        logger.exception(f"Curl fallback error for {url}")

    logger.warning(f"Persistent 403 for {url}, stopping retries.")
    return (
        f"Unable to access content at {url}. The website is blocking automated "
        f"access (403 Forbidden). This may be due to the site's terms of service "
        f"or security measures."
    )


async def _scrape_website_internal(
    url: str,
    timeout_seconds: int = 15,
    max_retries: int = 4,
    base_backoff_delay: float = 2.0,
    max_jitter: float = 1.0,
) -> str | None:
    """Internal implementation of website scraping."""
    # Validate URL first
    if not url_validator.is_valid_url(url):
        logger.error(f"Invalid or unsafe URL: {url}")
        return "Error: Invalid or unsafe URL. Only public HTTP/HTTPS URLs are allowed."

    # Check cache first
    if url in _scrape_cache and (time.time() - _scrape_cache[url]["timestamp"]) < CACHE_TTL_SECONDS:
        logger.info(f"Returning cached content for {url}")
        return _scrape_cache[url]["content"]

    logger.info(f"Attempting to scrape URL: {url}")

    auth_message = "This page requires authentication. Unable to access content."

    retries = 0
    while retries < max_retries:
        try:
            # Add delay with jitter
            await asyncio.sleep(secrets.SystemRandom().uniform(1.0, 3.0 + retries))

            # Choose headers based on retry count
            current_headers = ALTERNATIVE_HEADERS if retries % 2 == 1 else ENHANCED_HEADERS
            logger.debug(
                f"Attempt {retries + 1}/{max_retries} for {url} using "
                f"{'alternative' if retries % 2 == 1 else 'primary'} headers"
            )

            content, is_auth_required = await _try_aiohttp_scrape(url, timeout_seconds, current_headers)

            if is_auth_required:
                logger.warning(f"Authentication required detected for {url}")
                return _cache_and_return(url, auth_message)

            if content is not None:
                return _cache_and_return(url, content)

        except aiohttp.ClientResponseError as e:
            logger.exception(f"HTTP error {e.status} occurred for {url}")

            if e.status == HTTP_FORBIDDEN:
                result = await _handle_403_error(url, retries, max_retries)
                if result is not None:
                    return _cache_and_return(url, result)

        except (aiohttp.ClientError, TimeoutError, Exception) as e:
            error_type = type(e).__name__
            if isinstance(e, TimeoutError):
                logger.warning(f"Timeout occurred for {url} after {timeout_seconds} seconds")
            else:
                logger.exception(f"{error_type} exception for {url}")

        retries += 1
        if retries < max_retries:
            # Exponential backoff with jitter
            backoff_delay = base_backoff_delay * (2 ** (retries - 1))
            jitter = secrets.SystemRandom().uniform(-max_jitter, max_jitter)
            actual_delay = max(0, backoff_delay + jitter)
            logger.info(
                f"Retrying {url} in {actual_delay:.2f} seconds... (Attempt {retries}/{max_retries})"
            )
            await asyncio.sleep(actual_delay)

    return _cache_and_return(url, None)


def google_search(query: str, num_results: int = 10) -> list[dict[str, str]] | None:
    """Performs a Google Custom Search and returns results.

    Args:
        query: The search query string.
        num_results: The number of search results to return (max 10 for free CSE).

    Returns:
        A list of search result dictionaries (title, link, snippet), or None if an error occurs.
    """
    print(f"Starting Google search for query: '{query}'")
    print(
        f"Using API key: {GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-5:] if GOOGLE_API_KEY else None}"
    )
    print(f"Using CSE ID: {GOOGLE_CSE_ID}")
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("GOOGLE_API_KEY or GOOGLE_CSE_ID not found in environment variables.")
        return None
    try:
        print("Creating Google API service...")
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        print(f"Executing search with query='{query}', cx='{GOOGLE_CSE_ID}', num={num_results}")
        result = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
        print(f"Search completed. Result keys: {result.keys() if result else None}")
        search_items = result.get("items", [])
        print(f"Found {len(search_items)} search items")
        output_results = [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
            for item in search_items
        ]
    except HttpError as e:
        logger.exception(f"An API error occurred: {e.resp.status}")
        print(f"Full error details: {e!s}")
        return None
    except Exception as e:
        logger.exception("An unexpected error occurred during Google Search")
        print(f"Error type: {type(e).__name__}")
        print(f"Full error details: {e!s}")
        return None
    else:
        return output_results


def format_search_results(results: list[dict[str, str]]) -> str:
    """Format search results as a markdown string with improved readability.

    Args:
        results: List of search result dictionaries.

    Returns:
        Formatted markdown string with proper markdown links and structure.
    """
    if not results:
        return "No results found."

    output = "## Web Search Results\n\n"

    for i, res in enumerate(results, 1):
        title = res.get("title", "No title")
        link = res.get("link", "")
        snippet = res.get("snippet", "No description available")

        # Create a clean markdown link with proper formatting
        # Use the title as the link text, ensuring it's clean
        clean_title = title.replace("|", "-").strip()

        output += f"### {i}. [{clean_title}]({link})\n\n"
        output += f"{snippet}\n\n"

        # Add source domain for clarity
        if link:
            try:
                from urllib.parse import urlparse

                domain = urlparse(link).netloc
                output += f"*Source: {domain}*\n\n"
            except Exception:
                # Silently ignore URL parsing errors for source attribution
                logger.debug(f"Could not parse URL for source attribution: {link}")

        output += "---\n\n"

    return output


async def google_search_async(query: str, num_results: int = 10) -> list[dict[str, str]] | None:
    """Asynchronous wrapper for the Google search function.

    Args:
        query: The search query string.
        num_results: The number of search results to return (max 10 for free CSE).

    Returns:
        A list of search result dictionaries (title, link, snippet), or None if an error occurs.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: google_search(query, num_results))


async def _get_llm_decision(url: str, original_user_query: str) -> dict:
    """Get LLM decision for URL processing operation."""
    system_prompt = """Determine the best way to process a URL based on the user's query.
Choose one of the following operations:
1. FULL_TEXT_SCRAPE: If the user wants the general content of the page, or to perform their own analysis later.
2. SPECIFIC_INFO_EXTRACTION: If the user is looking for specific pieces of information (e.g.,
   contact details, product price, specific facts). If so, also specify the 'target_information'
   they are looking for.
3. PAGE_SUMMARIZATION: If the user wants a summary of the page content.
Respond with a JSON object indicating your choice. Examples:
For full text: {{"operation": "FULL_TEXT_SCRAPE"}}
For specific info: {{"operation": "SPECIFIC_INFO_EXTRACTION",
    "target_information": "the main contact email address"}}
For summarization: {{"operation": "PAGE_SUMMARIZATION"}}
JSON Response:"""
    user_prompt = (
        f"Given the URL '{url}' and the user's query: '{original_user_query}', "
        f"what operation should be performed?"
    )
    decision_prompt = utils.format_prompt(system_prompt, user_prompt)

    from persistent_llm_server import get_llm_server
    llm_server = await get_llm_server()

    raw_llm_output = await llm_server.generate(
        prompt=decision_prompt,
        max_tokens=200,
        temperature=0.3,
        session_id="web_scraper_decision",
    )

    # Extract JSON from response
    llm_response_text = raw_llm_output.strip()

    # Check for markdown code blocks
    if "```json" in llm_response_text:
        start = llm_response_text.find("```json") + 7
        end = llm_response_text.find("```", start)
        if end > start:
            llm_response_text = llm_response_text[start:end].strip()
        else:
            llm_response_text = llm_response_text[start:].strip()

    # Find JSON object boundaries
    start_idx = llm_response_text.find("{")
    end_idx = llm_response_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        llm_response_text = llm_response_text[start_idx : end_idx + 1]

    logger.debug(f"LLM Decision Raw Response for {url}: {llm_response_text}")

    try:
        decision_json = json.loads(llm_response_text)
    except json.JSONDecodeError:
        logger.exception(
            f"JSONDecodeError parsing LLM decision for {url}. "
            f"Raw text: '{llm_response_text}'"
        )
        # Fallback to full scrape
        return {"operation": "FULL_TEXT_SCRAPE"}
    else:
        return decision_json


async def _handle_full_text_scrape(url: str) -> str | None:
    """Handle full text scraping operation."""
    raw_content = await scrape_website(url)
    if raw_content:
        logger.info(f"Successfully scraped full text for {url} ({len(raw_content)} chars)")
        return raw_content
    else:
        logger.warning(f"Failed to scrape content for FULL_TEXT_SCRAPE on {url}")
        return None


async def _handle_specific_info_extraction(
    url: str, decision_json: dict, llm_client=None, model_lock=None
) -> str | None:
    """Handle specific information extraction operation."""
    target_info = decision_json.get("target_information")
    if not target_info:
        logger.warning(
            f"LLM decided SPECIFIC_INFO_EXTRACTION but no target_information for {url}"
        )
        return await scrape_website(url)

    logger.debug(f"Attempting SPECIFIC_INFO_EXTRACTION for '{target_info}' from {url}")
    raw_content = await scrape_website(url)
    if raw_content:
        extracted_info = await intelligent_extract_from_url(
            url=url,
            content=raw_content,
            user_query=f"Extract the following: {target_info}",
            llm_client=llm_client,
            model_lock=model_lock,
        )
        if extracted_info:
            logger.info(f"Successfully extracted '{target_info}' for {url}")
            return extracted_info
        else:
            logger.warning(
                f"Failed to extract '{target_info}' for {url}, "
                f"returning raw content snippet as fallback"
            )
            return raw_content[:1000]
    else:
        logger.warning(f"Failed to scrape content for SPECIFIC_INFO_EXTRACTION on {url}")
        return None


async def _handle_page_summarization(
    url: str, llm_client=None, model_lock=None
) -> str | None:
    """Handle page summarization operation."""
    logger.debug(f"Attempting PAGE_SUMMARIZATION for {url}")
    raw_content = await scrape_website(url)
    if raw_content:
        summary = await intelligent_extract_from_url(
            url=url,
            content=raw_content,
            user_query="Summarize this page content.",
            llm_client=llm_client,
            model_lock=model_lock,
        )
        if summary:
            logger.info(f"Successfully summarized {url}")
            return summary
        else:
            logger.warning(
                f"Failed to summarize {url}, returning raw content snippet as fallback"
            )
            return raw_content[:1000]
    else:
        logger.warning(f"Failed to scrape content for PAGE_SUMMARIZATION on {url}")
        return None


async def process_url_request(
    url: str, original_user_query: str, llm_client=None, model_lock=None
) -> Any | None:
    """Primary dispatcher for URL processing.

    Uses an LLM call to determine the type of web operation needed (e.g., full scrape, specific extraction)
    and then calls the appropriate internal function.

    Args:
        url: The URL to process.
        original_user_query: The user's original query related to this URL.
        llm_client: The initialized LLM client.
        model_lock: The priority lock for accessing the LLM.

    Returns:
        Processed content from the URL (e.g., full text, extracted info, summary), or None/error dict.
    """
    logger.debug(f"process_url_request: Starting processing for URL: {url}")
    logger.debug(f"Original user query context: {original_user_query[:100]}...")

    try:
        decision_json = await _get_llm_decision(url, original_user_query)
        operation = decision_json.get("operation")
        logger.debug(f"LLM Decided Operation for {url}: {operation}")

        if operation == "FULL_TEXT_SCRAPE":
            return await _handle_full_text_scrape(url)
        elif operation == "SPECIFIC_INFO_EXTRACTION":
            return await _handle_specific_info_extraction(url, decision_json, llm_client, model_lock)
        elif operation == "PAGE_SUMMARIZATION":
            return await _handle_page_summarization(url, llm_client, model_lock)
        else:
            logger.warning(
                f"Unknown operation '{operation}' decided by LLM for {url}. "
                f"Falling back to full scrape."
            )
            return await scrape_website(url)
    except Exception:
        logger.exception(f"Error in process_url_request for {url}")
        return await scrape_website(url)


async def intelligent_extract_from_url(
    url: str, content: str, user_query: str, llm_client=None, model_lock=None
) -> str | None:
    """Fetches a webpage, then uses an LLM to extract specific information based on an extraction query.

    Returns content formatted in markdown for better readability in the UI.

    Args:
        url: The URL to scrape.
        content: The content of the webpage.
        user_query: A natural language query describing what information to extract.
        llm_client: An LLM client object with a create_completion method.
        model_lock: Priority lock for accessing the LLM.

    Returns:
        The extracted information as a markdown-formatted string from the LLM, or None if an error occurs.
    """
    # LLM client and model_lock are now optional since we use persistent LLM server
    # Define the system prompt with markdown formatting instructions
    system_prompt = """You are Jane, an expert web content analyzer and formatter. Your task is to
extract information from web content based on the user's query.

🚨 CRITICAL LINK FORMATTING RULES - ABSOLUTE REQUIREMENTS 🚨

**COMPLETELY FORBIDDEN - NEVER USE THESE:**
- [REF]1[/REF] - FORBIDDEN
- [REF]2[/REF] - FORBIDDEN
- [REF]anything[/REF] - FORBIDDEN
- [REF]source[/REF] - FORBIDDEN
- [REF]url[/REF] - FORBIDDEN
- ANY variation of [REF]...[/REF] - COMPLETELY FORBIDDEN

**REQUIRED LINK FORMAT - ALWAYS USE THIS:**
- [Website Title](URL) - REQUIRED
- [CBS News](URL) - GOOD
- [Wikipedia](URL) - GOOD
- [White House](URL) - GOOD

**LINK TEXT RULES:**
- Use website title or domain name as link text, NEVER numbers
- NEVER use "1", "2", "3", or any numbers as link text
- Use descriptive titles like "CBS News", "Wikipedia", "White House", "CNN"
- If title is too long, use shortened version like "CBS News article", "Wikipedia page"
- NEVER use entire sentences or long phrases as link text

**EXAMPLES:**
❌ WRONG: "According to the source [REF]1[/REF]"
✅ CORRECT: "According to [CBS News](URL)"

❌ WRONG: "The data shows [REF]2,3[/REF]"
✅ CORRECT: "The data shows according to [Wikipedia](URL) and [White House](URL)"

🚨 MANDATORY TABLE FORMATTING RULES 🚨

**ALWAYS USE MARKDOWN TABLES FOR STRUCTURED DATA:**
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data A   | Data B   | Data C   |

**FORMATTING REQUIREMENTS:**
1. **ALWAYS use structured markdown formatting** - never plain text paragraphs for data
2. **MANDATORY TABLE USAGE**: When presenting any structured data, comparisons, lists of items with attributes, or multiple data points, YOU MUST use markdown tables
3. Use appropriate markdown elements:
   - # Headings and ## Subheadings for organization
   - * or - for bullet points and numbered lists for structured information
   - **TABLES ARE MANDATORY** for any structured data, stats, comparisons, or multi-attribute information
   - **bold** and *italic* for emphasis
   - ```language code blocks``` for any code or structured content
   - [descriptive links](URL) when referencing sources

**CONTENT RULES:**
- **STRUCTURE EVERYTHING**: Convert plain paragraphs into organized sections with headers, tables, and lists
- Include the most important information first
- Do not use phrases like "Here's what I found" or other unnecessary explanations
- Respond with ONLY the extracted information in well-formatted markdown with tables and structure
- When referencing the original source, use the format: [Source Title](URL)

🚨 FINAL REMINDER: NEVER use [REF] tags. ALWAYS use [Title](URL) format. 🚨"""
    # Define the user prompt with content and extraction query
    user_prompt = f"""Extraction Query: {user_query}
Web Content from {url} (may be truncated):
{content}
Please extract and format the relevant information using markdown."""
    # Use the standardized format_prompt function from utils
    formatted_prompt = utils.format_prompt(system_prompt, user_prompt)
    try:
        # Use persistent LLM server
        from persistent_llm_server import get_llm_server

        llm_server = await get_llm_server()

        # Extract the generated text response
        extracted_data = await llm_server.generate(
            prompt=formatted_prompt,
            max_tokens=512,
            temperature=0.7,
            session_id="web_scraper_extraction",
        )
        # Ensure the response has proper markdown formatting
        if extracted_data and not any(
            md_element in extracted_data for md_element in ["#", "-", "*", "|", "```", "**"]
        ):
            # If no markdown elements detected, add a heading with the URL as a fallback
            extracted_data = f"# Content from {url}\n\n{extracted_data}"
        # Add source link at the end if not already present
        if url not in extracted_data and extracted_data:
            extracted_data += f"\n\n---\n**Source**: [{url}]({url})"
            return extracted_data
        else:
            return extracted_data
    except Exception:
        logger.exception(f"Error during LLM completion call for intelligent extraction from {url}")
        return None
    finally:
        # Lock release is handled automatically by the context manager
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape text content from a website, perform a Google search, or "
        "intelligently process a URL using an LLM dispatcher."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="One or more URLs for basic text scraping.", nargs="+")
    group.add_argument(
        "-s",
        "--search",
        help="Perform a Google search with the specified query.",
        type=str,
        metavar="QUERY",
    )
    group.add_argument(
        "--process-url", help="URL to process with LLM-based task routing.", type=str, metavar="URL"
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="User query for intelligent processing or specific extraction from a URL "
        "(required if --process-url is used).",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for scraping (default: 10)",
    )
    parser.add_argument(
        "-n",
        "--num-results",
        type=int,
        default=5,
        help="Number of search results to return (default: 5) for Google search.",
    )
    args = parser.parse_args()

    # Mock LLM and Lock for CLI testing
    class MockLlamaClient:
        """Mock LLM client for CLI testing purposes."""

        def __init__(self, model_path=None, **kwargs):
            """Initialize mock LLM client."""
            print(
                f"MockLlamaClient initialized. Model path: {model_path if model_path else 'Not specified'}"
            )
            self.model_path = model_path

        def create_completion(self, prompt, **kwargs):
            """Create a mock completion response."""
            print("MockLlamaClient: create_completion called.")
            print(f"  Prompt: {prompt[:100]}...")
            print(f"  Kwargs: {kwargs}")
            # Simulate LLM decision for different scenarios
            if "web task routing assistant" in prompt.lower():
                # This is the decision prompt in process_url_request
                if "recipe" in prompt.lower() or "ingredients" in prompt.lower():
                    response_text = json.dumps(
                        {
                            "operation": "specific_extraction",
                            "parameters": {
                                "extraction_query": "List all ingredients and cooking steps"
                            },
                        }
                    )
                elif "summary" in prompt.lower() or "summarize" in prompt.lower():
                    response_text = json.dumps({"operation": "summarize_page", "parameters": {}})
                else:
                    response_text = json.dumps({"operation": "full_text_scrape", "parameters": {}})
            else:
                # This is for intelligent_extract_from_url
                response_text = (
                    f"Mock LLM extraction based on query: {args.query if args.query else 'Unknown Query'}\n\n"
                    f"Extracted from URL: {args.process_url}\n\n"
                    "This is simulated extracted content for testing purposes."
                )
            return {"choices": [{"text": response_text}]}

    class MockPriorityLock:
        """Mock priority lock for CLI testing purposes."""

        def __init__(self):
            """Initialize mock priority lock."""
            self._locked = False
            self._owner = None
            print("MockPriorityLock initialized.")

        async def acquire(self, priority=None, debug_name=None):
            """Acquire mock lock."""
            request_id = f"mock_req_{int(time.time() * 1000)}"
            print(f"MockPriorityLock: Acquire called by {debug_name} with priority {priority}")
            if self._locked:
                print(f"MockPriorityLock: Already locked by {self._owner}, waiting...")
                await asyncio.sleep(0.1)  # Simulate waiting
            self._locked = True
            self._owner = (request_id, debug_name)
            return request_id

        def release(self, request_id):
            """Release mock lock."""
            print(f"MockPriorityLock: Release called for {request_id}")
            if self._owner and self._owner[0] == request_id:
                self._locked = False
                self._owner = None
            else:
                print(f"MockPriorityLock: Warning - Request {request_id} doesn't own the lock")

    def _setup_stdout_buffering():
        """Configure stdout and stderr for immediate flushing."""
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)

    async def _handle_search_command():
        """Handle the search command functionality."""
        print(
            f"Performing Google Search for: '{args.search}' (max {args.num_results} results)...\n"
        )
        sys.stdout.flush()
        search_results = await google_search_async(
            args.search, num_results=args.num_results
        )
        if search_results:
            for i, res in enumerate(search_results, 1):
                print(f"Result {i}:")
                print(f"  Title: {res.get('title')}")
                print(f"  Link: {res.get('link')}")
                print(f"  Snippet: {res.get('snippet')}\n")
                sys.stdout.flush()
        else:
            print("No results found or an error occurred.", file=sys.stderr)
            sys.stderr.flush()
            sys.exit(1)

    async def _handle_url_scraping():
        """Handle URL scraping functionality."""
        all_successful = True
        for i, url_item in enumerate(args.url):
            if i > 0:
                print("\n" + "-" * 80 + "\n")  # Separator for multiple URLs
            print(f"Scraping {url_item} (basic text extraction)...")
            scraped_content = await scrape_website(url_item, args.timeout)
            if scraped_content:
                print(scraped_content)
            else:
                all_successful = False
                print(f"Failed to scrape {url_item}", file=sys.stderr)
        if not all_successful:
            sys.exit(1)

    async def _handle_process_url():
        """Handle URL processing with LLM functionality."""
        if not args.query:
            parser.error("--query is required when using --process-url")
        print(f"\nProcessing URL {args.process_url} with user query: '{args.query}'\n")
        print("Using LLM-powered dispatcher to determine the appropriate operation...\n")

        mock_llm = MockLlamaClient(model_path="mock_model_path")
        mock_lock = MockPriorityLock()

        result = await process_url_request(
            args.process_url, args.query, llm_client=mock_llm, model_lock=mock_lock
        )
        if result:
            print("\n--- Result from URL Processing ---")
            print(result)
            print("--- End Result ---")
        else:
            print("Failed to process URL.", file=sys.stderr)
            sys.exit(1)

    async def run_cli():
        """Run the command-line interface for web scraper."""
        _setup_stdout_buffering()
        print("\n=== Web Scraper CLI Started ===")

        try:
            if args.search:
                await _handle_search_command()
            elif args.url:
                await _handle_url_scraping()
            elif args.process_url:
                await _handle_process_url()
        except Exception as e:
            print(f"\nAn unexpected error occurred in run_cli: {e}", file=sys.stderr)
            print(f"Error type: {type(e).__name__}", file=sys.stderr)
            print(f"Error details: {e!s}", file=sys.stderr)
            sys.stderr.flush()
            sys.exit(1)

        print("\n=== Web Scraper CLI Completed ===")
        sys.stdout.flush()

    # Handle asyncio for Windows if needed
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Run the async CLI function
    asyncio.run(run_cli())


async def _check_for_url_in_query(query: str) -> str | None:
    """Check if query contains a URL and handle direct scraping."""
    import re

    url_pattern = r"https?://[^\s]+"
    urls_in_query = re.findall(url_pattern, query)

    if urls_in_query:
        url_to_scrape = urls_in_query[0]
        logger.info(f"Detected URL in query, switching to direct scraping: {url_to_scrape}")

        scraped_content = await scrape_website(url_to_scrape)
        if scraped_content:
            return f"# Content from {url_to_scrape}\n\n{scraped_content}"
        else:
            return f"Failed to scrape content from {url_to_scrape}"

    return None


async def _get_llm_search_decision(query: str, formatted_results: str) -> dict:
    """Get LLM decision on whether snippets are sufficient."""
    system_prompt = """You are Jane, evaluating whether Google search snippets contain enough information to answer a user's query.

🚨 CRITICAL LINK FORMATTING RULES - ABSOLUTE REQUIREMENTS 🚨

**COMPLETELY FORBIDDEN - NEVER USE THESE:**
- [REF]1[/REF] - FORBIDDEN
- [REF]2[/REF] - FORBIDDEN
- [REF]anything[/REF] - FORBIDDEN
- [REF]source[/REF] - FORBIDDEN
- [REF]url[/REF] - FORBIDDEN
- ANY variation of [REF]...[/REF] - COMPLETELY FORBIDDEN

**REQUIRED LINK FORMAT - ALWAYS USE THIS:**
- [Website Title](URL) - REQUIRED
- [CBS News](URL) - GOOD
- [Wikipedia](URL) - GOOD
- [White House](URL) - GOOD

**LINK TEXT REQUIREMENTS:**
- Use the website title or domain name as link text, NOT numbers
- NEVER use numbers like "1", "2", "3" as link text
- Use descriptive website titles like "CBS News", "Wikipedia", "White House", "CNN"
- If title is too long, use shortened version like "CBS News article", "Wikipedia page"
- NEVER use entire sentences or long phrases as link text
- Examples: [CBS News](URL), [Wikipedia](URL), [White House](URL), [CNN article](URL)

You must return your responses in proper markdown formatting and use markdown tables for structured data.

Analyze if these search snippets contain sufficient information to fully answer the user's query.
Respond with a JSON object:
{
    "sufficient": true/false,
    "reason": "Brief explanation",
    "urls_to_scrape": ["url1", "url2"] // Only if sufficient=false, max 3 URLs that would be most helpful
}"""

    user_prompt = f"""User Query: {query}

Search Results:
{formatted_results}"""

    formatted_prompt = format_prompt(system_prompt, user_prompt)

    from persistent_llm_server import get_llm_server
    llm_server = await get_llm_server()

    response_text = await llm_server.generate(
        prompt=formatted_prompt,
        max_tokens=200,
        temperature=0.1,
        session_id="web_search_decision",
    )
    logger.info(f"🤖 LLM raw response: {response_text}")

    if not response_text:
        logger.warning("LLM returned empty response, defaulting to snippets only")
        return {"sufficient": True}

    # Simple JSON extraction without regex
    json_text = response_text

    # Remove markdown code blocks if present
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end > start:
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text[start:].strip()
    else:
        # Look for raw JSON boundaries
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_text = response_text[start_idx : end_idx + 1]
        else:
            logger.warning(f"No JSON found in LLM response: {response_text[:200]}")
            return {"sufficient": True}

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON decision: {json_text}")
        return {"sufficient": True}


async def _scrape_selected_urls(urls_to_scrape: list, results: list) -> str:
    """Scrape selected URLs and format the results."""
    output_parts = ["\n\n## Detailed Content from Selected Results\n\n"]

    # Use semaphore to limit concurrent scrapes
    semaphore = asyncio.Semaphore(3)

    async def scrape_and_format(url):
        """Helper coroutine to scrape and format a single URL."""
        if not url:
            return None

        async with semaphore:
            try:
                # First validate URL accessibility before scraping
                logger.info(f"Validating URL accessibility: {url}")
                is_accessible = await validate_url_accessibility(url)
                if not is_accessible:
                    logger.warning(f"Skipping inaccessible URL: {url}")
                    return None

                logger.info(f"✅ URL accessible, starting scrape: {url}")
                content = await scrape_website(url=url, timeout_seconds=3, max_retries=1)
                logger.info(
                    f"🔍 Scrape completed for {url}: {len(content) if content else 0} chars"
                )

                if content and len(content.strip()) > MIN_CONTENT_LENGTH:
                    # Find the title from original results
                    title = "Unknown"
                    for result in results:
                        if result.get("link") == url:
                            title = result.get("title", "Unknown")
                            break

                    # Truncate very long content
                    if len(content) > CONTENT_PREVIEW_LIMIT:
                        content = content[:CONTENT_PREVIEW_LIMIT] + "\n\n[Content truncated...]"

                    return (
                        f"### Content from: {title}\nURL: {url}\n\n{content.strip()}\n\n---\n\n"
                    )
            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {e}")
                return None

    # Create tasks for concurrent scraping
    tasks = [scrape_and_format(url) for url in urls_to_scrape[:3]]  # Max 3 URLs

    # Run all scraping tasks concurrently
    logger.info(f"🚀 Starting concurrent scraping of {len(tasks)} URLs...")
    scraped_contents = await asyncio.gather(*tasks)
    logger.info(
        f"📊 Scraping completed. Results: {[type(c).__name__ if c else 'None' for c in scraped_contents]}"
    )

    # Process results
    scraped_count = 0
    for i, content in enumerate(scraped_contents):
        if content:
            output_parts.append(content)
            scraped_count += 1
            logger.info(f"✅ Added scraped content #{i + 1}: {len(content)} chars")
        else:
            logger.warning(f"❌ Scraped content #{i + 1} was None/empty")

    if scraped_count == 0:
        logger.warning("⚠️ Deep scraping failed completely")
        return ""
    else:
        logger.info(f"🎉 Successfully scraped {scraped_count}/{len(tasks)} URLs")
        return "".join(output_parts)


def perform_web_search(query: str, num_results: int = 5) -> str | None:
    """Perform a web search and return formatted results.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Formatted search results as string, or None if search fails
    """
    try:
        results = google_search(query, num_results)
        if results:
            return format_search_results(results)
        else:
            return None
    except Exception:
        logger.exception("Error performing web search")
        return None


async def perform_web_search_async(
    query: str,
    num_results: int = 5,
    llm_client=None,
    model_lock=None,
) -> str | None:
    """Intelligent async web search that uses LLM to determine if deep scraping is needed.

    First gets Google search results, then asks LLM if the snippets are sufficient
    to answer the query. Only performs deep scraping if LLM says it's needed.

    Args:
        query: Search query
        num_results: Number of results to return
        llm_client: Optional LLM client for intelligent decision making
        model_lock: Optional GPU lock for LLM inference

    Returns:
        Formatted search results, optionally with scraped content if LLM deems necessary
    """
    try:
        # Check if the query contains a URL - if so, scrape it directly instead of searching
        url_result = await _check_for_url_in_query(query)
        if url_result:
            return url_result

        # Regular search flow
        results = await google_search_async(query, num_results)
        if not results:
            return None

        # Format the search results
        formatted_results = format_search_results(results)

        # Ask LLM if snippets are sufficient
        try:
            decision = await _get_llm_search_decision(query, formatted_results)

            if decision.get("sufficient", True):
                logger.info(
                    f"📋 LLM: Snippets sufficient - {decision.get('reason', 'No reason provided')}"
                )
                result = formatted_results
            else:
                # LLM says we need to scrape
                logger.info(
                    f"🔍 LLM: Deep scraping needed - {decision.get('reason', 'No reason provided')}"
                )
                urls_to_scrape = decision.get("urls_to_scrape", [])

                if not urls_to_scrape:
                    # If no specific URLs provided, scrape top 3
                    urls_to_scrape = [r.get("link") for r in results[:3] if r.get("link")]

                # Perform deep scraping on selected URLs
                scraped_content = await _scrape_selected_urls(urls_to_scrape, results)

                if not scraped_content:
                    # If scraping failed, return snippets only
                    logger.warning("⚠️ Deep scraping failed completely, returning snippets only")
                    result = formatted_results
                else:
                    # Return combined snippets + scraped content
                    result = formatted_results + scraped_content
                    logger.info(f"📝 Final result length: {len(result)} chars")

        except Exception as e:
            logger.warning(f"LLM decision failed: {e}, defaulting to snippets only")
            result = formatted_results

    except Exception:
        logger.exception("Error performing async web search")
        return None
    else:
        return result
