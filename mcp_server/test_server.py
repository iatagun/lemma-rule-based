"""
MCP Server Test Suite

Test all available tools.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from mcp_server.server import (
    analyze_word,
    analyze_all_words,
    check_dictionary,
    get_project_info,
    test_suffixes,
)


async def run_tests():
    print("=" * 60)
    print("MCP Server Test Suite")
    print("=" * 60)

    # Test 1: Project Info
    print("\n1. Project Info:")
    result = await get_project_info({})
    print(result[0].text)

    # Test 2: Single Word Analysis
    print("\n2. Single Word Analysis (evlerinden):")
    result = await analyze_word({"word": "evlerinden", "upos": "NOUN"})
    print(result[0].text)

    # Test 3: Verb Analysis
    print("\n3. Verb Analysis (geliyorum):")
    result = await analyze_word({"word": "geliyorum", "upos": "VERB"})
    print(result[0].text)

    # Test 4: All Analyses
    print("\n4. All Analyses (gelirin):")
    result = await analyze_all_words({"word": "gelirin", "upos": "VERB", "max_results": 3})
    print(result[0].text)

    # Test 5: Dictionary Check
    print("\n5. Dictionary Check (kitap):")
    result = await check_dictionary({"word": "kitap", "find_root": True})
    print(result[0].text)

    # Test 6: Suffix Test
    print("\n6. Suffix Test (geldim):")
    result = await test_suffixes({"word": "geldim", "upos": "VERB"})
    print(result[0].text)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())
