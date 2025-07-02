#!/usr/bin/env python3
"""Advanced colored logging configuration for professional terminal output.
Provides intelligent color coding, structured formatting, and enhanced readability.
"""

import logging
import re
import sys
from datetime import UTC, datetime


class ColorCodes:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"

    # Text colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"

    # Text styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Extended color palette for memory operations
    MEMORY_NEURAL = "\033[38;5;129m"  # Purple for neural memory
    MEMORY_BASIC = "\033[38;5;33m"  # Blue for basic memory
    MEMORY_REDIS = "\033[38;5;196m"  # Red for Redis operations
    MEMORY_SAVE = "\033[38;5;46m"  # Bright green for saves
    MEMORY_RETRIEVE = "\033[38;5;226m"  # Yellow for retrieval
    MEMORY_DELETE = "\033[38;5;208m"  # Orange for deletion
    MEMORY_IMPORTANCE_HIGH = "\033[38;5;201m"  # Hot pink for high importance
    MEMORY_IMPORTANCE_MED = "\033[38;5;123m"  # Light blue for medium importance
    MEMORY_IMPORTANCE_LOW = "\033[38;5;250m"  # Gray for low importance

    # Gradient colors for memory importance (0.1-1.0)
    from typing import ClassVar

    IMPORTANCE_GRADIENT: ClassVar[dict] = {
        range(0, 3): "\033[38;5;244m",  # 0.0-0.2: Dark gray
        range(3, 5): "\033[38;5;250m",  # 0.3-0.4: Light gray
        range(5, 7): "\033[38;5;123m",  # 0.5-0.6: Light blue
        range(7, 8): "\033[38;5;33m",  # 0.7: Blue
        range(8, 9): "\033[38;5;226m",  # 0.8: Yellow
        range(9, 10): "\033[38;5;208m",  # 0.9: Orange
        range(10, 11): "\033[38;5;196m",  # 1.0: Red
    }


class IntelligentColorFormatter(logging.Formatter):
    """Intelligent logging formatter that applies colors based on:
    - Log level (ERROR, WARNING, INFO, DEBUG)
    - Content type (numbers, strings, technical terms)
    - Context (model parameters, file paths, etc.)
    """

    def __init__(self):
        super().__init__()

        # Level-based color mapping
        self.level_colors = {
            logging.DEBUG: ColorCodes.BRIGHT_BLACK + ColorCodes.DIM,
            logging.INFO: ColorCodes.BRIGHT_BLUE,
            logging.WARNING: ColorCodes.BRIGHT_YELLOW,
            logging.ERROR: ColorCodes.BRIGHT_RED,
            logging.CRITICAL: ColorCodes.RED + ColorCodes.BG_YELLOW + ColorCodes.BOLD,
        }

        # Component-based colors
        self.component_colors = {
            "BACKEND": ColorCodes.BRIGHT_MAGENTA,
            "FRONTEND": ColorCodes.BRIGHT_CYAN,
            "DATABASE": ColorCodes.BRIGHT_GREEN,
            "MODEL": ColorCodes.YELLOW,
            "API": ColorCodes.BLUE,
            "MEMORY": ColorCodes.MAGENTA,
            "NETWORK": ColorCodes.CYAN,
        }

        # Content-type colors
        self.content_colors = {
            "number": ColorCodes.BRIGHT_GREEN,
            "path": ColorCodes.CYAN,
            "url": ColorCodes.BRIGHT_CYAN,
            "technical": ColorCodes.YELLOW,
            "success": ColorCodes.GREEN,
            "error_detail": ColorCodes.RED,
            "parameter": ColorCodes.BRIGHT_MAGENTA,
            "value": ColorCodes.WHITE,
        }

    def colorize_content(self, message: str) -> str:
        """Apply intelligent content-based coloring to message text."""
        import re

        # File paths and directories
        message = re.sub(r"(/[^\s]+)", f"{ColorCodes.CYAN}\\1{ColorCodes.RESET}", message)

        # URLs
        message = re.sub(
            r"(https?://[^\s]+)",
            f"{ColorCodes.BRIGHT_CYAN}{ColorCodes.UNDERLINE}\\1{ColorCodes.RESET}",
            message,
        )

        # Numbers (integers and floats)
        message = re.sub(
            r"\b(\d+\.?\d*)\b", f"{ColorCodes.BRIGHT_GREEN}\\1{ColorCodes.RESET}", message
        )

        # Technical parameters (key=value pairs)
        message = re.sub(
            r"(\w+)(\s*=\s*)([^\s,]+)",
            f"{ColorCodes.BRIGHT_MAGENTA}\\1{ColorCodes.RESET}\\2"
            f"{ColorCodes.WHITE}\\3{ColorCodes.RESET}",
            message,
        )

        # Model/technical terms
        technical_terms = [
            "llama",
            "aria",
            "mistral",
            "token",
            "embedding",
            "ctx",
            "gpu",
            "cuda",
            "memory",
            "model",
            "inference",
            "attention",
            "transformer",
            "neural",
            "ai",
            "redis",
            "database",
            "cache",
            "session",
            "api",
            "endpoint",
        ]

        for term in technical_terms:
            message = re.sub(
                f"\\b({term})\\b",
                f"{ColorCodes.YELLOW}\\1{ColorCodes.RESET}",
                message,
                flags=re.IGNORECASE,
            )

        # Success indicators
        success_terms = [
            "success",
            "loaded",
            "initialized",
            "connected",
            "ready",
            "ok",
            "completed",
        ]
        for term in success_terms:
            message = re.sub(
                f"\\b({term})\\b",
                f"{ColorCodes.GREEN}\\1{ColorCodes.RESET}",
                message,
                flags=re.IGNORECASE,
            )

        # Error/warning indicators
        error_terms = ["error", "failed", "failure", "exception", "warning", "timeout", "abort"]
        for term in error_terms:
            message = re.sub(
                f"\\b({term})\\b",
                f"{ColorCodes.RED}\\1{ColorCodes.RESET}",
                message,
                flags=re.IGNORECASE,
            )

        return message

    def colorize_memory_operations(self, message: str) -> str:
        """Apply spectacular coloring to memory operations - like a brilliant painter! 🎨"""
        import re

        # Memory operation type detection and coloring
        memory_patterns = {
            # Neural Memory Operations - Purple gradient
            r"(🧠.*?neural.*?memory|neural.*?memory.*?🧠)": ColorCodes.MEMORY_NEURAL
            + ColorCodes.BOLD,
            r"(\[NEURAL.*?\])": ColorCodes.MEMORY_NEURAL,
            # Basic Memory Operations - Blue gradient
            r"(🔄.*?basic.*?memory|basic.*?memory.*?🔄)": ColorCodes.MEMORY_BASIC + ColorCodes.BOLD,
            r"(\[MEMORY.*?\])": ColorCodes.MEMORY_BASIC,
            # Redis Operations - Red gradient
            r"(redis_utils.*?INFO.*?Successfully)": ColorCodes.MEMORY_REDIS + ColorCodes.BOLD,
            r"(memory_b:.*?)": ColorCodes.MEMORY_REDIS,
            # Memory Save Operations - Bright Green
            r"(\[MEMORY SAVE.*?\])": ColorCodes.MEMORY_SAVE + ColorCodes.BOLD,
            r"(Added memory:)": ColorCodes.MEMORY_SAVE + ColorCodes.BOLD,
            r"(Successfully added/updated)": ColorCodes.MEMORY_SAVE,
            # Memory Retrieval Operations - Yellow
            r"(\[MEMORY RETRIEVAL.*?\])": ColorCodes.MEMORY_RETRIEVE + ColorCodes.BOLD,
            r"(Retrieved.*?memories)": ColorCodes.MEMORY_RETRIEVE,
            r"(get_relevant_memories|get_context_memories)": ColorCodes.MEMORY_RETRIEVE,
            # Batch Operations - Cyan with progress bars
            r"(Batches:.*?\d+%.*?)": ColorCodes.BRIGHT_CYAN + ColorCodes.BOLD,
        }

        # Apply all patterns except importance scores
        for pattern, color in memory_patterns.items():
            message = re.sub(pattern, f"{color}\\1{ColorCodes.RESET}", message, flags=re.IGNORECASE)

        # Handle importance scores separately with gradient coloring
        import re

        importance_matches = list(re.finditer(r"(\(importance:\s*(\d+\.?\d*)\))", message))
        for match in reversed(importance_matches):  # Reverse to maintain positions
            colored_importance = self._get_importance_color(match)
            message = message[: match.start()] + colored_importance + message[match.end() :]

        # Memory content highlighting - make the actual memory text stand out
        message = re.sub(
            r"('.*?')(.*?)(\(importance:)",
            f"{ColorCodes.BRIGHT_WHITE + ColorCodes.ITALIC}\\1{ColorCodes.RESET}\\2\\3",
            message,
        )

        # Add beautiful icons for different operations
        message = re.sub(r"\[MEMORY SAVE", f"💾 {ColorCodes.MEMORY_SAVE}[MEMORY SAVE", message)
        message = re.sub(
            r"\[MEMORY RETRIEVAL", f"🔍 {ColorCodes.MEMORY_RETRIEVE}[MEMORY RETRIEVAL", message
        )
        message = re.sub(r"\[NEURAL", f"🧠 {ColorCodes.MEMORY_NEURAL}[NEURAL", message)

        # Add progress bar styling for batches
        message = re.sub(
            r"(Batches:)\s*(\d+%)\|([█▉▊▋▌▍▎▏ ]*)\|\s*(\d+/\d+)",
            (
                f"{ColorCodes.BRIGHT_CYAN}\\1{ColorCodes.RESET} "
                f"{ColorCodes.BRIGHT_GREEN}\\2{ColorCodes.RESET}|"
                f"{ColorCodes.BRIGHT_BLUE}\\3{ColorCodes.RESET}| "
                f"{ColorCodes.BRIGHT_YELLOW}\\4{ColorCodes.RESET}"
            ),
            message,
        )

        return message

    def _get_importance_color(self, match) -> str:
        """Get gradient color based on importance score."""
        importance_text = match.group(1)
        score_text = match.group(2)

        try:
            score = float(score_text)
            score_int = int(score * 10)  # Convert 0.1-1.0 to 1-10

            # Find the right color range
            for score_range, color in ColorCodes.IMPORTANCE_GRADIENT.items():
                if score_int in score_range:
                    return f"{color + ColorCodes.BOLD}{importance_text}{ColorCodes.RESET}"

            # Fallback to high importance color
            return (
                f"{ColorCodes.MEMORY_IMPORTANCE_HIGH + ColorCodes.BOLD}"
                f"{importance_text}{ColorCodes.RESET}"
            )

        except ValueError:
            return importance_text

    def format_timestamp(self, record: logging.LogRecord) -> str:
        """Format timestamp with subtle coloring."""
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        return f"{ColorCodes.BRIGHT_BLACK}[{timestamp}]{ColorCodes.RESET}"

    def format_level(self, record: logging.LogRecord) -> str:
        """Format log level with appropriate colors and styling."""
        level_color = self.level_colors.get(record.levelno, ColorCodes.WHITE)
        level_name = record.levelname

        # Add padding for alignment
        padded_level = level_name.ljust(8)

        return f"{level_color}{padded_level}{ColorCodes.RESET}"

    def format_logger_name(self, record: logging.LogRecord) -> str:
        """Format logger name with component-based coloring."""
        logger_name = record.name

        # Extract component type from logger name
        component_color = ColorCodes.BRIGHT_WHITE

        for component, color in self.component_colors.items():
            if component.lower() in logger_name.lower():
                component_color = color
                break

        # Truncate long logger names
        if len(logger_name) > 20:
            logger_name = "..." + logger_name[-17:]

        return f"{component_color}{logger_name.ljust(20)}{ColorCodes.RESET}"

    def format(self, record: logging.LogRecord) -> str:
        """Main formatting method that combines all elements."""
        # Format components
        timestamp = self.format_timestamp(record)
        level = self.format_level(record)
        logger_name = self.format_logger_name(record)

        # Get the raw message
        message = record.getMessage()

        # Apply memory-specific coloring FIRST (most important!)
        message = self.colorize_memory_operations(message)

        # Then apply general content coloring
        message = self.colorize_content(message)

        # Combine with separators
        separator = f"{ColorCodes.BRIGHT_BLACK}│{ColorCodes.RESET}"

        formatted = (
            f"{timestamp} {separator} {level} {separator} {logger_name} {separator} {message}"
        )

        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


class StreamFormatter(IntelligentColorFormatter):
    """Formatter specifically optimized for model streaming output."""

    def format(self, record: logging.LogRecord) -> str:
        """Special formatting for streaming-related logs."""
        message = record.getMessage()

        # Special handling for batch operations (these are memory-related)
        if "batches:" in message.lower():
            timestamp = self.format_timestamp(record)
            # Apply beautiful batch coloring
            message = self.colorize_memory_operations(message)
            return f"{timestamp} {ColorCodes.BRIGHT_CYAN}⚡{ColorCodes.RESET} {message}"

        # Special handling for streaming messages
        if any(keyword in message.lower() for keyword in ["streaming", "token", "chunk", "sse"]):
            timestamp = self.format_timestamp(record)

            # Use a more compact format for streaming
            if "token" in message.lower():
                # Highlight streaming tokens
                message = re.sub(
                    r"(token[^:]*:)\s*(['\"][^'\"]*['\"])",
                    f"{ColorCodes.BRIGHT_YELLOW}\\1{ColorCodes.RESET} "
                    f"{ColorCodes.GREEN}\\2{ColorCodes.RESET}",
                    message,
                    flags=re.IGNORECASE,
                )

                return f"{timestamp} {ColorCodes.BRIGHT_BLUE}▶{ColorCodes.RESET} {message}"

            elif "chunk" in message.lower():
                return f"{timestamp} {ColorCodes.CYAN}●{ColorCodes.RESET} {self.colorize_content(message)}"

            elif "streaming" in message.lower():
                return f"{timestamp} {ColorCodes.MAGENTA}⟩{ColorCodes.RESET} {self.colorize_content(message)}"

        # Fall back to standard formatting (which now includes beautiful memory coloring!)
        return super().format(record)


def setup_colored_logging(level: int = logging.INFO, enable_stream_formatting: bool = True) -> None:
    """Set up intelligent colored logging for the entire application.

    Args:
        level: Logging level (default: INFO)
        enable_stream_formatting: Whether to use special formatting for streaming logs
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    # Choose formatter based on whether we're in a TTY
    if sys.stdout.isatty():
        if enable_stream_formatting:
            formatter = StreamFormatter()
        else:
            formatter = IntelligentColorFormatter()
    else:
        # Use plain formatter for non-TTY (like redirected output)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # Set specific levels for different components
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    print(f"{ColorCodes.GREEN}✓ Intelligent colored logging initialized{ColorCodes.RESET}")


def create_section_separator(title: str, width: int = 80) -> str:
    """Create a visually appealing section separator."""
    title_with_spaces = f" {title} "
    padding = width - len(title_with_spaces)
    left_padding = padding // 2
    right_padding = padding - left_padding

    return (
        f"{ColorCodes.BRIGHT_BLUE}"
        f"{'═' * left_padding}{title_with_spaces}{'═' * right_padding}"
        f"{ColorCodes.RESET}"
    )


def log_startup_banner(app_name: str, version: str = "1.0.0") -> None:
    """Log a beautiful startup banner."""
    logger = logging.getLogger(__name__)

    banner = f"""
{ColorCodes.BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║    {ColorCodes.BRIGHT_WHITE}{app_name.center(64)}{ColorCodes.BRIGHT_CYAN}    ║
║                                                                          ║
║    {ColorCodes.YELLOW}Version: {version.ljust(58)}{ColorCodes.BRIGHT_CYAN}    ║
║    {ColorCodes.GREEN}Status:  Initializing...{" " * 47}{ColorCodes.BRIGHT_CYAN}    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝{ColorCodes.RESET}
"""

    print(banner)
    logger.info(f"Starting {app_name} v{version}")


if __name__ == "__main__":
    # Demo of the colored logging system
    setup_colored_logging()

    logger = logging.getLogger("demo")

    log_startup_banner("Mistral Small Chat Application", "v3.2-24B")

    logger.debug("This is a debug message with /path/to/file and value=123")
    logger.info("Model loaded successfully with ctx_length=131072 and embedding_dim=3840")
    logger.warning("High memory usage detected: 85% of available GPU memory")
    logger.error("Failed to connect to Redis at localhost:6379")
    logger.critical("CRITICAL: Model initialization failed!")

    # Streaming demo
    stream_logger = logging.getLogger("streaming")
    stream_logger.info("Starting streaming response for session_id abc123")
    stream_logger.info("Token received: 'Hello'")
    stream_logger.info("Chunk #1 processed with 42 tokens")
    stream_logger.info("Streaming completed successfully")

    # Memory operations demo - Show off the brilliant colors! 🎨
    memory_logger = logging.getLogger("memory.memory_manager")
    redis_logger = logging.getLogger("redis_utils")

    print(f"\n{ColorCodes.BRIGHT_WHITE}🎨 Memory Operations Color Demo:{ColorCodes.RESET}")

    # Neural memory operations
    memory_logger.info("🧠 Retrieving neural memories for session session_123")
    memory_logger.info("[NEURAL MEMORY] Context retrieved: 2048 characters")

    # Basic memory operations
    memory_logger.info("🔄 Falling back to basic memory retrieval for session session_123")
    memory_logger.info("[MEMORY RETRIEVAL unknown] Retrieved 5 memories for user session_123")

    # Batch operations with progress
    import logging

    batch_logger = logging.getLogger("sentence_transformers")
    batch_logger.info("Batches:   0%|          | 0/1 [00:00<?, ?it/s]")
    batch_logger.info("Batches: 100%|██████████| 1/1 [00:01<00:00,  1.23it/s]")
