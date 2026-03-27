# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# WebBrowserTool - Interactive web browsing agent
# Based on patterns from browser-use/browser-use (84K stars)
#
# Uses:
# - smolagents CodeAgent for agent orchestration
# - Helium/Selenium for browser automation
# - nanoGPT API with Mistral Small 4 for LLM inference
#
# Key design decisions (evidence-based):
# - Default to DuckDuckGo (fewer captchas) - browser-use pattern
# - No fallback chain - let agent choose engine
# - URL encoding with urllib.parse.quote_plus
# - Google uses &udm=14 to disable AI overview
# - Return raw page text, let LLM parse results

from __future__ import annotations

import logging
import mimetypes
import os
import time
import urllib.parse
import uuid
from typing import Any, Optional

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# smolagents imports
from smolagents import CodeAgent, OpenAIModel, tool

# Helium for browser automation
import helium
from helium import (
    click,
    write,
    scroll_down,
    scroll_up,
    Text,
    Link,
    start_chrome,
)

from kgot.utils import UsageStatistics
from kgot.utils.log_and_statistics import collect_stats

logger = logging.getLogger("Controller.WebBrowserTool")

# Global browser driver
_driver = None


def init_browser():
    """Initialize the Helium browser with anti-detection settings."""
    global _driver
    if _driver is None:
        from selenium import webdriver
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--force-device-scale-factor=1")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        _driver = start_chrome(headless=True, options=chrome_options)
    return _driver


def get_driver():
    """Get the browser driver."""
    global _driver
    if _driver is None:
        init_browser()
    return _driver


# =============================================================================
# Browser Tools (smolagents @tool decorated)
# Based on browser-use/browser-use patterns
# =============================================================================

@tool
def search_web(query: str, engine: str = "duckduckgo") -> str:
    """
    Search the web using DuckDuckGo (default) or Google.

    Use this tool when:
    - Finding information on any topic
    - Getting current information on recent events
    - Looking for specific websites or documentation

    Args:
        query: The search query. Be specific and use natural language.
        engine: "duckduckgo" (default, fewer captchas) or "google".

    Returns:
        Search results page content.

    Tips:
        - Use "site:edu" for academic sources
        - Add year for current info: "AI trends 2025"
        - Be specific: "FastAPI dependency injection tutorial" not just "FastAPI"

    Examples:
        GOOD: "site:arxiv.org transformer attention mechanism 2024"
        GOOD: "Neo4j vs FalkorDB graph database comparison"
        AVOID: Single words like "AI" or "python"
    """
    driver = get_driver()
    encoded_query = urllib.parse.quote_plus(query)

    # Search URLs - Google uses udm=14 to disable AI overview
    search_urls = {
        'duckduckgo': f'https://duckduckgo.com/?q={encoded_query}',
        'google': f'https://www.google.com/search?q={encoded_query}&udm=14',
    }

    if engine.lower() not in search_urls:
        return f"Unknown engine '{engine}'. Use 'duckduckgo' or 'google'."

    search_url = search_urls[engine.lower()]

    try:
        helium.go_to(search_url)
        time.sleep(1.5)
        body = driver.find_element("tag name", "body")
        return f"Search results from {engine}:\n\n{body.text[:5000]}"
    except Exception as e:
        return f"Search failed on {engine}: {str(e)}. Try a different query or engine."


@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for encyclopedic information.

    Use this tool when:
    - Need factual, well-sourced information
    - Getting background on concepts, people, places
    - Looking for established knowledge (not recent events)

    Args:
        query: The topic or concept to search for.

    Returns:
        Wikipedia summary with title, extract, and URL.

    Tips:
        - Works best for established, well-documented topics
        - May not have info on very recent events or niche topics
        - Good starting point before web search for current info
    """
    try:
        # Search for page
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit=1&format=json"
        search_resp = requests.get(search_url, timeout=10)

        if not search_resp.json()[1]:
            return f"No Wikipedia page found for '{query}'. Try a different search term."

        page_title = search_resp.json()[1][0]

        # Get summary
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
        summary_resp = requests.get(summary_url, timeout=10)

        if summary_resp.status_code != 200:
            return f"Could not fetch Wikipedia summary for '{page_title}'"

        data = summary_resp.json()
        title = data.get('title', page_title)
        extract = data.get('extract', 'No summary available')
        page_url = data.get('content_urls', {}).get('desktop', {}).get('page', 'N/A')

        return f"**Wikipedia: {title}**\n\n{extract[:800]}\n\nURL: {page_url}"

    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


@tool
def download_file(url: str) -> str:
    """
    Download a file from a URL.

    Use this tool when:
    - Need to download PDF, Excel, PowerPoint, or other files
    - File needs local inspection via text inspector tool

    Args:
        url: Direct URL to the file.

    Returns:
        Path to downloaded file and size information.

    Supported types: .pdf, .xlsx, .pptx, .docx, .wav, .mp3, .png, .jpg

    Tips:
        - Use after finding file links via search_web
        - Downloaded files can be inspected with text inspector tool
    """
    try:
        # Handle arxiv URLs
        if "arxiv.org/abs/" in url:
            url = url.replace("/abs/", "/pdf/")

        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        # Detect file type
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type) or ".download"

        # Generate filename
        filename = f"file_{uuid.uuid4().hex[:8]}{extension}"
        download_dir = os.path.join(os.getcwd(), "kgot", "tools", "downloads")
        os.makedirs(download_dir, exist_ok=True)

        filepath = os.path.join(download_dir, filename)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(filepath)
        return f"Downloaded: {filepath}\nFile size: {file_size} bytes\nType: {content_type}"

    except requests.exceptions.Timeout:
        return f"Download failed: Request timed out. Try a smaller file or check the URL."
    except requests.exceptions.RequestException as e:
        return f"Download failed: {str(e)}"
    except Exception as e:
        return f"Download failed: {str(e)}"


@tool
def navigate_to(url: str) -> str:
    """
    Navigate to a specific URL.

    Use this tool when:
    - You have a specific URL from search results
    - Need to visit a known website directly

    Args:
        url: The URL to navigate to (must include http:// or https://).

    Returns:
        The page content after navigation.

    Tips:
        - Use after search_web to visit specific results
        - URL must be complete with http:// or https://
    """
    driver = get_driver()
    try:
        helium.go_to(url)
        time.sleep(1.5)
        body = driver.find_element("tag name", "body")
        return f"Navigated to: {url}\n\n{body.text[:5000]}"
    except Exception as e:
        return f"Navigation failed: {str(e)}"


@tool
def click_element(text: str) -> str:
    """
    Click on a clickable element by its visible text.

    Use this tool when:
    - Need to click buttons, links, or other interactive elements
    - Following navigation flows on websites

    Args:
        text: The visible text on the element to click.

    Returns:
        Result of the click action.

    Tips:
        - Use exact text from the page
        - Works for buttons, links, and other clickable elements
    """
    try:
        if Text(text).exists():
            click(Text(text))
            time.sleep(1)
            return f"Clicked element with text: '{text}'"
        elif Link(text).exists():
            click(Link(text))
            time.sleep(1)
            return f"Clicked link with text: '{text}'"
        else:
            return f"Element with text '{text}' not found"
    except Exception as e:
        return f"Click failed: {str(e)}"


@tool
def type_text(text: str, into: Optional[str] = None) -> str:
    """
    Type text into an input field.

    Use this tool when:
    - Filling out forms
    - Entering search queries in website search boxes
    - Inputting text in any text field

    Args:
        text: The text to type.
        into: Optional label or placeholder of the input field.

    Returns:
        Result of typing action.

    Tips:
        - Use 'into' parameter to specify which field when multiple exist
        - For search boxes, the placeholder text often works as 'into'
    """
    try:
        if into:
            write(text, into=into)
        else:
            write(text)
        return f"Typed: '{text}'"
    except Exception as e:
        return f"Type failed: {str(e)}"


@tool
def scroll_page(direction: str = "down", pixels: int = 500) -> str:
    """
    Scroll the page up or down.

    Use this tool when:
    - Reading long pages
    - Looking for content below the fold
    - Navigating through paginated content

    Args:
        direction: "up" or "down" (default: "down").
        pixels: Number of pixels to scroll (default: 500).

    Returns:
        Current viewport content after scrolling.
    """
    driver = get_driver()
    if direction.lower() == "down":
        scroll_down(pixels)
    else:
        scroll_up(pixels)
    time.sleep(0.5)
    body = driver.find_element("tag name", "body")
    return f"Scrolled {direction} {pixels} pixels\n\n{body.text[:3000]}"


@tool
def find_on_page(text: str) -> str:
    """
    Find text on the current page (like Ctrl+F).

    Use this tool when:
    - Looking for specific information on a long page
    - Verifying if certain content exists on the page

    Args:
        text: The text to search for on the page.

    Returns:
        Whether the text was found and count of matches.
    """
    driver = get_driver()
    try:
        from selenium.webdriver.common.by import By
        elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
        if elements:
            elem = elements[0]
            driver.execute_script("arguments[0].scrollIntoView(true);", elem)
            return f"Found {len(elements)} matches for '{text}'. Scrolled to first match."
        return f"Text '{text}' not found on page"
    except Exception as e:
        return f"Find failed: {str(e)}"


@tool
def close_popup() -> str:
    """
    Close any visible popup or modal by pressing Escape.

    Use this tool when:
    - Popups or modals block the page content
    - Cookie consent dialogs appear
    - Need to dismiss overlay elements

    Returns:
        Result of closing action.
    """
    from selenium.webdriver.common.keys import Keys
    from selenium import webdriver
    driver = get_driver()
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    return "Pressed Escape to close popup"


@tool
def get_current_page_content() -> str:
    """
    Get the full text content of the current page.

    Use this tool when:
    - Need to extract all visible text from a page
    - Reading article or documentation content
    - Gathering comprehensive page information

    Returns:
        The visible text content of the page.
    """
    driver = get_driver()
    body = driver.find_element("tag name", "body")
    return body.text[:8000]


@tool
def get_page_url() -> str:
    """
    Get the current page URL.

    Use this tool when:
    - Need to verify current location
    - Storing or sharing the current URL

    Returns:
        The current URL.
    """
    driver = get_driver()
    return driver.current_url


@tool
def go_back() -> str:
    """
    Navigate back to the previous page.

    Use this tool when:
    - Need to return to a previous page
    - Navigating through browser history

    Returns:
        The previous page content.
    """
    driver = get_driver()
    driver.back()
    time.sleep(1)
    body = driver.find_element("tag name", "body")
    return f"Went back to: {driver.current_url}\n\n{body.text[:3000]}"


# =============================================================================
# WebBrowserTool Schema (LangChain compatible)
# =============================================================================

class WebBrowserToolSchema(BaseModel):
    query: str = Field(
        description="A natural language request for what you want to find or do on the web. "
        "Include specific details like URLs, dates, or exact phrases when known. "
        "Example: 'Find the official documentation page for FastAPI dependency injection' "
        "or 'Go to github.com/huggingface/transformers and find the latest release version'"
    )


# =============================================================================
# WebBrowserTool (LangChain BaseTool wrapper)
# =============================================================================

class WebBrowserTool(BaseTool):
    """
    Interactive web browser controlled by an AI agent.

    The agent can navigate to URLs, click elements, fill forms, scroll pages,
    search the web, download files, and extract text content from web pages.
    Uses Helium/Selenium for automation and Mistral Small 4 via nanoGPT for reasoning.
    """

    name: str = "browser_use_tool"
    description: str = """
Interactive web browser agent. Navigate, search, click, and extract web content.

Use this tool when you need to:
- Search the web (uses DuckDuckGo by default - fewer captchas)
- Navigate to specific URLs and extract content
- Click elements, fill forms, scroll pages
- Download files (PDF, Excel, etc.)
- Search Wikipedia for encyclopedic info

This tool gives you CONTROL over a headless Chrome browser. The AI agent will:
1. Open a browser and navigate to URLs
2. Interact with page elements (click, type, scroll)
3. Extract and return relevant information

Parameters:
- query: Natural language request describing what to find/do on the web

Examples:
- "Search for the latest Python 3.13 release notes"
- "Go to github.com/owner/repo and find installation instructions"
- "Download the PDF from arxiv.org/abs/2301.12345"
- "Find Wikipedia summary of 'knowledge graph'"

DO NOT use for:
- Simple facts you already know (use your knowledge)
- Deep multi-source research (use deep_research_tool instead)
- Files on disk (use inspect_file_as_text tool)
"""

    args_schema: type[BaseModel] = WebBrowserToolSchema
    agent: CodeAgent = None
    usage_statistics: UsageStatistics = None

    def __init__(
        self,
        model_name: str = "mistralai/mistral-small-4-119b-2603",
        temperature: float = 0.5,
        usage_statistics: UsageStatistics = None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.usage_statistics = usage_statistics

        # Get nanoGPT API key
        nanogpt_api_key = os.getenv("NANOGPT_API_KEY", "")
        nanogpt_base_url = os.getenv("NANOGPT_BASE_URL", "https://nano-gpt.com/api/subscription/v1")

        # Initialize browser
        init_browser()

        # Create OpenAI-compatible model for nanoGPT
        model = OpenAIModel(
            model_id=model_name,
            api_base=nanogpt_base_url,
            api_key=nanogpt_api_key,
            temperature=temperature,
        )

        # Create CodeAgent with browser tools
        self.agent = CodeAgent(
            tools=[
                search_web,
                search_wikipedia,
                download_file,
                navigate_to,
                click_element,
                type_text,
                scroll_page,
                find_on_page,
                close_popup,
                get_current_page_content,
                get_page_url,
                go_back,
            ],
            model=model,
            max_steps=12,
        )

    @collect_stats("WebBrowserTool._run")
    def _run(self, query: str) -> str:
        """Execute the browsing agent."""
        try:
            # Add structured output instructions
            enhanced_query = f"""
You've been submitted this request: '{query}'

Help answer this question by browsing the web. Provide as much detail as possible.

Your final answer MUST contain these parts:
### 1. Search outcome (short version):
### 2. Search outcome (extremely detailed version):
### 3. Additional context:

If you cannot find the answer, explain what you tried and what was missing.

IMPORTANT: Default to DuckDuckGo for searches (fewer captchas). Use Google only if needed.
"""
            result = self.agent.run(enhanced_query)
            return str(result)
        except Exception as e:
            logger.error(f"WebBrowserTool error: {e}")
            return f"Browser agent encountered an error: {str(e)}\n\nPlease try rephrasing your request."


# =============================================================================
# Convenience functions and backward compatibility
# =============================================================================

def create_web_browser_tool(
    model_name: str = "mistralai/mistral-small-4-119b-2603",
    temperature: float = 0.5,
    usage_statistics: UsageStatistics = None,
) -> WebBrowserTool:
    """Create a WebBrowserTool instance with specified configuration."""
    return WebBrowserTool(
        model_name=model_name,
        temperature=temperature,
        usage_statistics=usage_statistics,
    )

# Backward compatibility aliases
BrowserUseTool = WebBrowserTool
create_browser_use_tool = create_web_browser_tool