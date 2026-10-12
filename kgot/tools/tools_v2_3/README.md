# Tool Set v2.3

This directory contains the tool set in the version 2.3.

Please note that we provide a [JSON file](additional_config_tools.json), which can be used to supply API keys and other configuration parameters to the respective tools.
Copy `additional_config_tools.template.json` to set these values; empty values
are skipped so plain environment variables keep working.

## Tools

| Tool name | Class | Key needed |
|-----------|-------|------------|
| `brightdata_search` | `BrightdataSearchTool` | `BRIGHT_DATA_API_TOKEN`, `BRIGHTDATA_SERP_ZONE` (default `serp`) |
| `brightdata_fetch` | `BrightdataFetchTool` | `BRIGHT_DATA_API_TOKEN`, `BRIGHTDATA_UNLOCKER_ZONE` (default `unlocker`) |
| `wikipedia_search` | `WikipediaSearchTool` | none |
| `hackernews_search` | `HackerNewsSearchTool` | none |
| `youtube_search` | `YouTubeSearchTool` | `YOUTUBE_API_KEY` (or `YOUTUBE_DATA_API_KEY`) |
| `youtube_transcript` | `YouTubeTranscriptTool` | none |
| `arxiv_search` | `ArxivSearchTool` | none |
| `openalex_search` | `OpenAlexSearchTool` | optional `OPENALEX_API_KEY`, optional `CONTACT_EMAIL` |
| `semantic_scholar_search` | `SemanticScholarSearchTool` | optional `SEMANTIC_SCHOLAR_API_KEY` |

Shared HTTP behavior (retries on transient failures only, per-host courtesy
delays, secret redaction, public-URL validation) lives in `http_utils.py`.
