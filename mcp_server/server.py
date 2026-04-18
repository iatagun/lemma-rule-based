"""
MCP Server for Turkish Morphological Analyzer

Provides tools for:
- Running benchmarks
- Analyzing words
- Testing functionality
- Accessing project documentation
"""

import sys
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

from morphology import (
    create_default_analyzer,
    TurkishDictionary,
    AnalysisFormatter,
)
from morphology.analyzer import MorphemeAnalysis
from morphology.phonology import turkish_lower

# Server instance
server = Server("turkish-morphology")

# Global analyzer instance
_analyzer = None
_dictionary = None


def get_analyzer():
    """Get or create the analyzer instance."""
    global _analyzer, _dictionary
    if _analyzer is None:
        dict_path = PROJECT_ROOT / "turkish_words.txt"
        _dictionary = TurkishDictionary.from_file(dict_path)
        _analyzer = create_default_analyzer(dictionary_path=str(dict_path))
    return _analyzer


# ── Tools ────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="analyze_word",
            description="Analyze a Turkish word and return its morphological decomposition",
            inputSchema={
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The Turkish word to analyze"
                    },
                    "upos": {
                        "type": "string",
                        "description": "Universal POS tag (NOUN, VERB, ADJ, etc.)",
                        "enum": ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM", "AUX", "ADP", "CCONJ", "PRON", "DET", "PART", "SCONJ", "INTJ"]
                    }
                },
                "required": ["word"]
            }
        ),
        Tool(
            name="analyze_all",
            description="Get all possible morphological analyses for a word",
            inputSchema={
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The Turkish word to analyze"
                    },
                    "upos": {
                        "type": "string",
                        "description": "Universal POS tag"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["word"]
            }
        ),
        Tool(
            name="run_benchmark",
            description="Run the BOUN Treebank benchmark and return results",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary_only": {
                        "type": "boolean",
                        "description": "Return only the summary (no detailed errors)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="check_dictionary",
            description="Check if a word exists in the Turkish dictionary",
            inputSchema={
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The word to check"
                    },
                    "find_root": {
                        "type": "boolean",
                        "description": "Also try to find the root (with morfophonemic rules)",
                        "default": False
                    }
                },
                "required": ["word"]
            }
        ),
        Tool(
            name="format_analysis",
            description="Format a morphological analysis as human-readable text",
            inputSchema={
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The original word"
                    },
                    "stem": {
                        "type": "string",
                        "description": "The stem"
                    },
                    "suffixes": {
                        "type": "string",
                        "description": "JSON string of suffixes [(form, label), ...]"
                    }
                },
                "required": ["word", "stem"]
            }
        ),
        Tool(
            name="test_suffixes",
            description="Test suffix matching for a word",
            inputSchema={
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The word to test"
                    },
                    "upos": {
                        "type": "string",
                        "description": "POS tag"
                    }
                },
                "required": ["word"]
            }
        ),
        Tool(
            name="get_project_info",
            description="Get general information about the project",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="load_skill",
            description="Load a skill file for the agent context",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill name to load: overview, phonology, benchmarking, adding_suffixes, debugging, suffix_reference, dep_parsing, morphology, all",
                        "enum": ["overview", "phonology", "benchmarking", "adding_suffixes", "debugging", "coding_standards", "suffix_reference", "dep_parsing", "morphology", "all"]
                    },
                    "category": {
                        "type": "boolean",
                        "description": "If True and skill_name is 'morphology' or 'all', load entire category",
                        "default": False
                    },
                    "preview": {
                        "type": "boolean",
                        "description": "If True, show only first 30 lines",
                        "default": False
                    }
                },
                "required": ["skill_name"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict[str, Any]
) -> list[TextContent]:
    """Handle tool calls."""

    if name == "analyze_word":
        return await analyze_word(arguments)

    elif name == "analyze_all":
        return await analyze_all_words(arguments)

    elif name == "run_benchmark":
        return await run_benchmark(arguments)

    elif name == "check_dictionary":
        return await check_dictionary(arguments)

    elif name == "format_analysis":
        return await format_analysis(arguments)

    elif name == "test_suffixes":
        return await test_suffixes(arguments)

    elif name == "get_project_info":
        return await get_project_info(arguments)

    elif name == "load_skill":
        return await load_skill(arguments)

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Tool Implementations ────────────────────────────────────────


async def analyze_word(args: dict) -> list[TextContent]:
    """Analyze a single word."""
    word = args["word"]
    upos = args.get("upos")

    analyzer = get_analyzer()
    result = analyzer.analyze(word, upos=upos)

    output = format_result(word, result)
    return [TextContent(type="text", text=output)]


async def analyze_all_words(args: dict) -> list[TextContent]:
    """Analyze a word and return all possible analyses."""
    word = args["word"]
    upos = args.get("upos")
    max_results = args.get("max_results", 5)

    analyzer = get_analyzer()
    results = analyzer.analyze_all(word, upos=upos, max_results=max_results)

    lines = [f"Word: {word} (POS: {upos or 'unknown'})\n"]
    lines.append(f"Found {len(results)} analyses:\n")

    for i, result in enumerate(results, 1):
        lines.append(f"  {i}. {format_result(word, result)}")

    return [TextContent(type="text", text="\n".join(lines))]


async def run_benchmark(args: dict) -> list[TextContent]:
    """Run the benchmark."""
    summary_only = args.get("summary_only", True)

    # Import benchmark module
    from benchmark.evaluate import parse_conllu, evaluate

    analyzer = get_analyzer()
    test_path = PROJECT_ROOT / "benchmark" / "test.conllu"

    if not test_path.exists():
        return [TextContent(type="text", text=f"Error: {test_path} not found")]

    tokens = parse_conllu(str(test_path))
    results = evaluate(tokens, analyzer)

    output = generate_benchmark_report(results, summary_only)
    return [TextContent(type="text", text=output)]


async def check_dictionary(args: dict) -> list[TextContent]:
    """Check dictionary for a word."""
    word = turkish_lower(args["word"])
    find_root = args.get("find_root", False)

    analyzer = get_analyzer()
    dictionary = analyzer._dictionary

    if dictionary is None:
        return [TextContent(type="text", text="Dictionary not loaded")]

    in_dict = dictionary.contains(word)

    lines = [f"Word: {word}"]
    lines.append(f"In dictionary: {in_dict}")

    if find_root:
        root = dictionary.find_root(word)
        lines.append(f"Root (find_root): {root}")

    return [TextContent(type="text", text="\n".join(lines))]


async def format_analysis(args: dict) -> list[TextContent]:
    """Format an analysis result."""
    word = args["word"]
    stem = args["stem"]
    suffixes_str = args.get("suffixes", "[]")

    import json
    try:
        suffixes = json.loads(suffixes_str)
    except:
        suffixes = []

    result = MorphemeAnalysis(stem=stem, suffixes=suffixes)

    formatter = AnalysisFormatter()
    output = formatter.format_analysis(word, result)

    return [TextContent(type="text", text=output)]


async def test_suffixes(args: dict) -> list[TextContent]:
    """Test which suffixes match a word."""
    word = args["word"]
    upos = args.get("upos")

    analyzer = get_analyzer()
    results = analyzer.analyze_all(word, upos=upos, max_results=3)

    lines = [f"Suffix test for: {word}\n"]

    for i, result in enumerate(results, 1):
        lines.append(f"  {i}. stem='{result.stem}'")
        for form, label in result.suffixes:
            lines.append(f"     └── {form} ({label})")

    return [TextContent(type="text", text="\n".join(lines))]


async def get_project_info(args: dict) -> list[TextContent]:
    """Get project information."""
    dict_path = PROJECT_ROOT / "turkish_words.txt"
    analyzer = get_analyzer()

    info = [
        "Turkish Morphological Analyzer",
        "=" * 40,
        f"Project: {PROJECT_ROOT.name}",
        "",
        "Statistics:",
        f"  Dictionary: {len(analyzer._dictionary._words) if analyzer._dictionary else 'N/A'} words",
        f"  Suffixes: {len(analyzer._registry.suffixes)} forms",
        "",
        "Available tools:",
        "  - analyze_word: Analyze a single word",
        "  - analyze_all: Get all possible analyses",
        "  - run_benchmark: Run BOUN benchmark",
        "  - check_dictionary: Check dictionary",
        "  - test_suffixes: Test suffix matching",
        "  - format_analysis: Format analysis result",
    ]

    return [TextContent(type="text", text="\n".join(info))]


# ── Helpers ─────────────────────────────────────────────────────


def format_result(word: str, result: MorphemeAnalysis) -> str:
    """Format a single analysis result."""
    parts = result.parts
    stem = result.stem
    lemma = result.lemma or result.root or stem

    lines = [
        f"Word: {word}",
        f"Stem: {stem}",
        f"Lemma: {lemma}",
        "Parts: " + " + ".join(parts),
    ]

    if result.suffixes:
        lines.append("Suffixes:")
        for form, label in result.suffixes:
            lines.append(f"  {form} → {label}")

    return "\n".join(lines)


def generate_benchmark_report(results: dict, summary_only: bool) -> str:
    """Generate benchmark report text."""
    lines = [
        "BOUN Treebank Benchmark Results",
        "=" * 40,
        f"Total: {results['total']}",
        f"Correct: {results['correct']}",
        f"Accuracy: {results['correct']/results['total']*100:.1f}%",
        "",
        "POS Breakdown:",
    ]

    by_pos = results.get("by_pos", {})
    for pos in sorted(by_pos.keys()):
        data = by_pos[pos]
        rate = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        lines.append(f"  {pos:8} {data['correct']:4}/{data['total']:4} ({rate:.1f}%)")

    if not summary_only and results.get("errors"):
        lines.append("")
        lines.append("Top Errors:")
        error_counts = {}
        for err in results["errors"]:
            key = f"{err['word']} → {err['predicted']} (gold: {err['gold']})"
            error_counts[key] = error_counts.get(key, 0) + 1

        for key, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {count}x {key}")

    return "\n".join(lines)


async def load_skill(args: dict) -> list[TextContent]:
    """Load a skill file."""
    skill_name = args["skill_name"]
    category = args.get("category", False)
    preview = args.get("preview", False)

    # Import skills module
    sys.path.insert(0, str(PROJECT_ROOT))
    from skills import load_skill as _load_skill

    content = _load_skill(skill_name, category=category)

    if preview:
        lines = content.split("\n")
        content = "\n".join(lines[:30]) + "\n... (preview)"

    return [TextContent(type="text", text=content)]


# ── Main ────────────────────────────────────────────────────────


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
