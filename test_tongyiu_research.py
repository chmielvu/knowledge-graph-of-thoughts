"""
Test script for TongyiuDeepResearch components.
Run from the project root directory.
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_configuration():
    """Test configuration module."""
    from kgot.tools.tools_v2_3.TongyiuDeepResearch.configuration import (
        DeepResearchConfig, SearchAPI
    )

    config = DeepResearchConfig.from_env()
    print(f"  Config created: max_research_loops={config.max_research_loops}")
    print(f"  SearXNG URLs: {config.searxng_urls[:2]}")
    print(f"  Fallback to DDG: {config.fallback_to_duckduckgo}")
    print("  [OK] Configuration")
    return config

def test_state():
    """Test state module."""
    from kgot.tools.tools_v2_3.TongyiuDeepResearch.state import (
        ResearchState, Source
    )

    state = ResearchState(research_topic="Test topic")
    print(f"  State created: topic={state.research_topic}")
    print(f"  Should continue (initial): {state.should_continue()}")
    print("  [OK] State")
    return state

def test_tools():
    """Test tools module (imports only)."""
    from kgot.tools.tools_v2_3.TongyiuDeepResearch.tools import (
        ResilientSearchTool, WikipediaTool, create_search_tools
    )
    print("  [OK] Tools imports")

def test_agent():
    """Test agent module (imports only)."""
    from kgot.tools.tools_v2_3.TongyiuDeepResearch.agent import (
        create_research_agent, run_research
    )
    print("  [OK] Agent imports")

def main():
    print("Testing TongyiuDeepResearch components...\n")

    try:
        test_configuration()
    except Exception as e:
        print(f"  [FAIL] Configuration: {e}")
        return 1

    try:
        test_state()
    except Exception as e:
        print(f"  [FAIL] State: {e}")
        return 1

    try:
        test_tools()
    except Exception as e:
        print(f"  [FAIL] Tools: {e}")
        return 1

    try:
        test_agent()
    except Exception as e:
        print(f"  [FAIL] Agent: {e}")
        return 1

    print("\nAll tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())