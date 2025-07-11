#!/usr/bin/env python3
"""Utility functions for consistent prompt formatting and other shared functionality.

This module provides a single source of truth for how prompts should be formatted
for the Mistral-Small-3.2-24B-Instruct-2506 LLM model. By centralizing the prompt format here, we ensure consistency
across all parts of the application and make it easier to update the prompt format
if needed in the future.
The prompt format follows the official Mistral instruction format:
<s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]
"""

import logging
from datetime import UTC

from gpu_lock import Priority

# Set up module logger
logger = logging.getLogger(__name__)


def format_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format system prompt and user prompt in the standardized format for Mistral-Small-3.2-24B-Instruct-2506 LLM.

    This is the single source of truth for how prompts should be formatted
    for the Mistral model. Any changes to the prompt format should be made here
    and will be reflected throughout the application.

    Args:
        system_prompt: The system instructions/prompt
        user_prompt: The user's query or message
    Returns:
        Formatted prompt string with correct special tokens
    """
    return f"<s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]"


def format_prompt_with_history(
    system_prompt: str, user_prompt: str, conversation_history: str = ""
) -> str:
    """Format system prompt, conversation history, and user prompt in the standardized format for Mistral-Small-3.2-24B-Instruct-2506 LLM.

    This variant includes conversation history in the prompt format. It ensures
    consistent formatting across the application when conversation history needs
    to be included.

    Args:
        system_prompt: The system instructions/prompt
        user_prompt: The user's query or message
        conversation_history: Optional formatted conversation history
    Returns:
        Formatted prompt string with correct special tokens including conversation history
    """
    return f"<s>[SYSTEM_PROMPT]{system_prompt}\n\n{conversation_history}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]"


class AIQueryRouter:
    """100% AI-driven query routing and analysis."""

    def __init__(self):
        """Initialize the AI query router."""
        pass  # No hardcoded patterns!

    def analyze_query_with_ai(self, query: str, llm=None, model_lock=None) -> dict:
        """Use the LLM to intelligently analyze and route user queries.

        Args:
            query: The user's input query
            llm: The language model instance
            model_lock: Lock for model access

        Returns:
            Dictionary with AI-determined routing information
        """
        if not llm or not model_lock:
            return {"error": "AI query analysis service temporarily unavailable."}

        system_prompt = """You are Jane, an intelligent query router. Analyze user queries and
determine what type of data they need.

You are not allowed to use [REF]URL[/REF] tags for any reason and must only use markdown links every time: [Text Here](URL)

CRITICAL INSTRUCTIONS:
1. Return ONLY a JSON object with your analysis
2. Determine if the query needs: cryptocurrency data, stock market data, web search, or
   general conversation
3. If crypto/stock data is needed, extract the specific symbols/coins mentioned
4. Be intelligent about synonyms, abbreviations, and context

JSON Response Format:
{
    "query_type": "crypto" | "stocks" | "web_search" | "general",
    "confidence": 0.0-1.0,
    "symbols": ["list", "of", "symbols"],
    "reasoning": "brief explanation of your analysis"
}

Examples:
- "bitcoin price" → {"query_type": "crypto", "confidence": 0.95, "symbols": ["bitcoin"],
  "reasoning": "User asking for Bitcoin price data"}
- "AAPL vs MSFT" → {"query_type": "stocks", "confidence": 0.9, "symbols": ["AAPL", "MSFT"],
  "reasoning": "Comparing Apple and Microsoft stocks"}
- "latest news about AI" → {"query_type": "web_search", "confidence": 0.8, "symbols": [],
  "reasoning": "User wants current news requiring web search"}
- "how are you today" → {"query_type": "general", "confidence": 0.95, "symbols": [],
  "reasoning": "General conversational query"}"""

        user_prompt = f"Analyze this user query: {query}"

        try:
            import asyncio
            import json

            async def get_ai_analysis():
                formatted_prompt = format_prompt(system_prompt, user_prompt)

                async with model_lock.acquire_context(
                    priority=Priority.HIGH,  # Query routing is critical
                    timeout=15.0,
                    debug_name="ai_query_routing",
                ):
                    response = llm.create_completion(
                        prompt=formatted_prompt,
                        max_tokens=None,
                        temperature=0.7,  # Recommended for Mistral
                        top_p=0.95,
                        min_p=0.0,
                        top_k=64,
                        stream=True,
                        echo=False,
                        stop=["[/INST]", "[/SYSTEM_PROMPT]"],
                    )

                    result_text = response["choices"][0]["text"].strip()

                    # Extract JSON from response without regex
                    start_idx = result_text.find("{")
                    end_idx = result_text.rfind("}")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        try:
                            analysis = json.loads(result_text[start_idx : end_idx + 1])

                            # Validate the response structure
                            if "query_type" in analysis:
                                return {
                                    "query_type": analysis.get("query_type", "general"),
                                    "confidence": analysis.get("confidence", 0.5),
                                    "symbols": analysis.get("symbols", []),
                                    "reasoning": analysis.get("reasoning", ""),
                                    "needs_crypto": (analysis.get("query_type") == "crypto"),
                                    "needs_stock": analysis.get("query_type") == "stocks",
                                    "needs_search": analysis.get("query_type") == "web_search",
                                    "original_query": query,
                                }
                        except json.JSONDecodeError:
                            pass

                    # If AI response can't be parsed, return error
                    return {
                        "error": f"Unable to analyze query. AI response: {result_text[:100]}..."
                    }

            # Run the async function
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(get_ai_analysis(), loop).result(timeout=15)
            else:
                return asyncio.run(get_ai_analysis())

        except Exception as e:
            return {"error": f"Query analysis failed: {e!s}"}

    def extract_crypto_symbols_with_ai(self, query: str, llm=None, model_lock=None) -> list:
        """Extract cryptocurrency symbols using AI inference."""
        if not llm or not model_lock:
            return []

        system_prompt = """You are Jane, a cryptocurrency expert. Your task is to analyze user queries
and extract relevant cryptocurrency coin IDs that should be fetched for price data.

You are not allowed to use [REF]URL[/REF] tags for any reason and must only use markdown links every time: [Text Here](URL)

CRITICAL INSTRUCTIONS:
1. Return ONLY a JSON array of CoinGecko coin IDs (lowercase, dash-separated format)
2. Use exact CoinGecko coin IDs: bitcoin, ethereum, binancecoin, ripple, cardano, solana,
   dogecoin, matic-network, avalanche-2, chainlink, etc.
3. If no specific coins are mentioned, return popular/trending coins
4. Maximum 10 coins
5. NO explanations, NO other text, ONLY the JSON array

Examples:
- "bitcoin price" → ["bitcoin"]
- "btc and eth" → ["bitcoin", "ethereum"]
- "crypto market" → ["bitcoin", "ethereum", "binancecoin", "ripple", "cardano"]
- "top altcoins" → ["ethereum", "binancecoin", "cardano", "solana", "dogecoin"]"""

        user_prompt = f"Extract cryptocurrency coin IDs from this query: {query}"

        try:
            import asyncio

            async def get_ai_symbols():
                formatted_prompt = format_prompt(system_prompt, user_prompt)

                async with model_lock.acquire_context(
                    priority=Priority.LOW, timeout=10.0, debug_name="crypto_symbol_extraction"
                ):
                    response = llm.create_completion(
                        prompt=formatted_prompt,
                        max_tokens=None,
                        temperature=0.7,  # Recommended for Mistral
                        top_p=0.95,
                        top_k=64,
                        min_p=0.0,
                        echo=False,
                        stream=False,
                        stop=["[/INST]", "[/SYSTEM_PROMPT]"],
                    )

                    result_text = response["choices"][0]["text"].strip()

                    # Extract JSON array from response without regex
                    import json

                    # Try to find JSON array in the response
                    start_idx = result_text.find("[")
                    end_idx = result_text.rfind("]")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        try:
                            symbols = json.loads(result_text[start_idx : end_idx + 1])
                            if isinstance(symbols, list):
                                return symbols[:10]  # Limit to 10
                        except json.JSONDecodeError:
                            pass

                    # If AI can't parse the response properly, return empty list
                    logger.warning("Unable to parse cryptocurrency symbols from AI response")
                    return []

            # Run the async function
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context
                return asyncio.run_coroutine_threadsafe(get_ai_symbols(), loop).result(timeout=10)
            else:
                return asyncio.run(get_ai_symbols())

        except Exception:
            logger.exception("AI crypto symbol extraction failed")
            # Return empty list on error
            return []

    def extract_stock_symbols_with_ai(self, query: str, llm=None, model_lock=None) -> list:
        """Extract stock symbols using AI inference."""
        if not llm or not model_lock:
            return []

        system_prompt = """You are Jane, a stock market expert. Your task is to analyze user queries and
extract relevant stock ticker symbols that should be fetched for price data.

You are not allowed to use [REF]URL[/REF] tags for any reason and must only use markdown links every time: [Text Here](URL)

CRITICAL INSTRUCTIONS:
1. Return ONLY a JSON array of valid US stock ticker symbols (uppercase)
2. Use standard NYSE/NASDAQ symbols: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, etc.
3. If no specific stocks are mentioned, return popular/major stocks
4. Maximum 10 stocks
5. NO explanations, NO other text, ONLY the JSON array

Examples:
- "apple stock" → ["AAPL"]
- "MSFT and GOOGL" → ["MSFT", "GOOGL"]
- "tech stocks" → ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
- "market leaders" → ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]"""

        user_prompt = f"Extract stock ticker symbols from this query: {query}"

        try:
            import asyncio

            async def get_ai_symbols():
                formatted_prompt = format_prompt(system_prompt, user_prompt)

                async with model_lock.acquire_context(
                    priority=Priority.LOW, timeout=10.0, debug_name="stock_symbol_extraction"
                ):
                    response = llm.create_completion(
                        prompt=formatted_prompt,
                        max_tokens=None,
                        temperature=0.3,  # More conservative for symbol extraction
                        top_p=0.95,
                        top_k=64,
                        min_p=0.0,
                        stream=False,
                        echo=False,
                        stop=["[/INST]", "[/SYSTEM_PROMPT]"],
                    )

                    result_text = response["choices"][0]["text"].strip()

                    # Extract JSON array from response without regex
                    import json

                    # Try to find JSON array in the response
                    start_idx = result_text.find("[")
                    end_idx = result_text.rfind("]")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        try:
                            symbols = json.loads(result_text[start_idx : end_idx + 1])
                            if isinstance(symbols, list):
                                return symbols[:10]  # Limit to 10
                        except json.JSONDecodeError:
                            pass

                    # If AI can't parse the response properly, return empty list
                    logger.warning("Unable to parse stock symbols from AI response")
                    return []

            # Run the async function
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, use create_task
                return asyncio.run_coroutine_threadsafe(get_ai_symbols(), loop).result(timeout=10)
            else:
                return asyncio.run(get_ai_symbols())

        except Exception:
            logger.exception("AI stock symbol extraction failed")
            # Return empty list on error
            return []


def get_timestamp() -> str:
    """Get current timestamp in ISO format.

    Returns:
        Current timestamp as ISO format string
    """
    from datetime import datetime

    return datetime.now(UTC).isoformat()


class AIDataProcessor:
    """100% AI-driven data processing and formatting."""

    def __init__(self):
        """Initialize the AI data processor."""
        pass

    def process_data_with_ai(
        self, raw_data: str, user_query: str, data_type: str, llm=None, model_lock=None
    ) -> str:
        """Let the AI intelligently process and format data based on the user's specific question.

        Args:
            raw_data: Raw data (JSON, tables, etc.)
            user_query: Original user question
            data_type: Type of data (crypto, stocks, web_search)
            llm: Language model instance
            model_lock: Model access lock

        Returns:
            AI-formatted response
        """
        if not llm or not model_lock:
            return f"Data processing service unavailable. Raw data: {raw_data[:500]}..."

        system_prompt = f"""You are Jane, a financial data analyst and expert communicator. Your task is to
process {data_type} data and present it in the most helpful way for the user's specific question.

You are not allowed to use [REF]URL[/REF] tags for any reason and must only use markdown links every time: [Text Here](URL)

CRITICAL INSTRUCTIONS:
1. Analyze the user's question to understand what they really want to know
2. Process the raw data intelligently - don't just display tables
3. Provide insights, trends, comparisons, and context where relevant
4. Use markdown formatting for maximum readability
5. Include source attributions where provided
6. Be conversational but informative
7. If data shows interesting patterns, highlight them
8. Always end with a brief summary or key takeaway

User wants to know about: {data_type} data
Raw data provided: This contains current market information

Remember: The user asked a specific question. Answer THAT question using the data, don't just
dump the data."""

        user_prompt = f"""User Question: {user_query}

Raw Data:
{raw_data}

Please analyze this data and provide a comprehensive, insightful response to the user's question."""

        try:
            import asyncio

            async def get_ai_response():
                formatted_prompt = format_prompt(system_prompt, user_prompt)

                async with model_lock.acquire_context(
                    priority=Priority.MEDIUM, timeout=20.0, debug_name="ai_data_processing"
                ):
                    response = llm.create_completion(
                        prompt=formatted_prompt,
                        max_tokens=None,  # Allow for detailed analysis
                        temperature=1.0,  # Slightly more creative for insights
                        top_p=0.95,
                        top_k=64,
                        min_p=0.0,
                        stream=False,
                        echo=False,
                        stop=["[/INST]", "[/SYSTEM_PROMPT]"],
                    )

                    result_text = response["choices"][0]["text"].strip()
                    return result_text

            # Run the async function
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(get_ai_response(), loop).result(timeout=20)
            else:
                return asyncio.run(get_ai_response())

        except Exception as e:
            return f"Unable to process data intelligently: {e!s}\n\nRaw data: {raw_data}"
