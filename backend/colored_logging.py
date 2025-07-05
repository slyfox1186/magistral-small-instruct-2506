#!/usr/bin/env python3
"""Advanced colored logging configuration for professional terminal output.
Provides intelligent color coding, structured formatting, and enhanced readability.
"""

import logging
import sys
from datetime import UTC, datetime

UTC = UTC


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

        # Pre-compile regex patterns for performance
        self._compile_regex_patterns()

    def _compile_regex_patterns(self):
        """Pre-compile all regex patterns for better performance."""
        # Technical terms for simple string matching
        self.technical_terms = [
            "llama", "aria", "mistral", "token", "embedding", "ctx",
            "gpu", "cuda", "memory", "model", "inference", "attention",
            "transformer", "neural", "ai", "redis", "database", "cache",
            "session", "api", "endpoint"
        ]

        # Success terms for simple string matching
        self.success_terms = [
            "success", "loaded", "initialized", "connected",
            "ready", "ok", "completed"
        ]

        # Error terms for simple string matching
        self.error_terms = [
            "error", "failed", "failure", "exception",
            "warning", "timeout", "abort"
        ]

        # Memory operation strings and their colors
        self.memory_strings = [
            ("🧠", "neural", "memory", ColorCodes.MEMORY_NEURAL + ColorCodes.BOLD),
            ("[NEURAL", ColorCodes.MEMORY_NEURAL),
            ("🔄", "basic", "memory", ColorCodes.MEMORY_BASIC + ColorCodes.BOLD),
            ("[MEMORY", ColorCodes.MEMORY_BASIC),
            ("redis_utils", "INFO", "Successfully", ColorCodes.MEMORY_REDIS + ColorCodes.BOLD),
            ("memory_b:", ColorCodes.MEMORY_REDIS),
            ("[MEMORY SAVE", ColorCodes.MEMORY_SAVE + ColorCodes.BOLD),
            ("Added memory:", ColorCodes.MEMORY_SAVE + ColorCodes.BOLD),
            ("Successfully added/updated", ColorCodes.MEMORY_SAVE),
            ("[MEMORY RETRIEVAL", ColorCodes.MEMORY_RETRIEVE + ColorCodes.BOLD),
            ("Retrieved", "memories", ColorCodes.MEMORY_RETRIEVE),
            ("get_relevant_memories", ColorCodes.MEMORY_RETRIEVE),
            ("get_context_memories", ColorCodes.MEMORY_RETRIEVE),
            ("Batches:", "%", ColorCodes.BRIGHT_CYAN + ColorCodes.BOLD),
        ]

    def colorize_content(self, message: str) -> str:
        """Apply intelligent content-based coloring to message text."""
        # Split message into words for easier processing
        words = message.split()
        colored_words = []

        for word in words:
            colored_word = word

            # Simple path detection (starts with /)
            if word.startswith('/') and len(word) > 1:
                colored_word = f"{ColorCodes.CYAN}{word}{ColorCodes.RESET}"

            # Simple URL detection (starts with http:// or https://)
            elif word.startswith(('http://', 'https://')):
                colored_word = f"{ColorCodes.BRIGHT_CYAN}{ColorCodes.UNDERLINE}{word}{ColorCodes.RESET}"

            # Simple number detection
            elif self._is_number(word):
                colored_word = f"{ColorCodes.BRIGHT_GREEN}{word}{ColorCodes.RESET}"

            # Simple key=value detection
            elif '=' in word and not word.startswith('=') and not word.endswith('='):
                parts = word.split('=', 1)
                if len(parts) == 2:
                    key, value = parts
                    colored_word = f"{ColorCodes.BRIGHT_MAGENTA}{key}{ColorCodes.RESET}={ColorCodes.WHITE}{value}{ColorCodes.RESET}"

            # Check for technical terms
            else:
                word_lower = word.lower()
                for term in self.technical_terms:
                    if term in word_lower:
                        colored_word = f"{ColorCodes.YELLOW}{word}{ColorCodes.RESET}"
                        break

                # Check for success terms
                if colored_word == word:
                    for term in self.success_terms:
                        if term in word_lower:
                            colored_word = f"{ColorCodes.GREEN}{word}{ColorCodes.RESET}"
                            break

                # Check for error terms
                if colored_word == word:
                    for term in self.error_terms:
                        if term in word_lower:
                            colored_word = f"{ColorCodes.RED}{word}{ColorCodes.RESET}"
                            break

            colored_words.append(colored_word)

        return ' '.join(colored_words)

    def _is_number(self, s: str) -> bool:
        """Check if a string is a number."""
        # Remove trailing punctuation
        s = s.rstrip('.,;:!?')
        try:
            float(s)
            return True
        except ValueError:
            return False

    def colorize_memory_operations(self, message: str) -> str:
        """Apply spectacular coloring to memory operations - like a brilliant painter! 🎨"""
        # Apply memory string patterns
        for item in self.memory_strings:
            if len(item) == 2:
                # Single string pattern
                string_to_find, color = item
                if string_to_find in message:
                    # Find and color the portion after the match
                    parts = message.split(string_to_find, 1)
                    if len(parts) == 2:
                        # Find the end of the relevant portion (usually until newline or certain chars)
                        end_idx = len(parts[1])
                        for end_char in ['\n', ',', '.', '|']:
                            idx = parts[1].find(end_char)
                            if idx != -1 and idx < end_idx:
                                end_idx = idx

                        colored_part = parts[1][:end_idx]
                        rest = parts[1][end_idx:]
                        message = parts[0] + f"{color}{string_to_find}{colored_part}{ColorCodes.RESET}" + rest

            elif len(item) == 4:
                # Multi-string pattern
                str1, str2, str3, color = item
                if str1 in message and str2 in message.lower() and str3 in message.lower():
                    # Simple approach: color the whole line containing all three strings
                    lines = message.split('\n')
                    for i, line in enumerate(lines):
                        if str1 in line and str2 in line.lower() and str3 in line.lower():
                            lines[i] = f"{color}{line}{ColorCodes.RESET}"
                    message = '\n'.join(lines)

        # Handle importance scores with simple string search
        if "(importance:" in message:
            parts = message.split("(importance:")
            new_parts = [parts[0]]
            for i in range(1, len(parts)):
                part = parts[i]
                # Find the closing parenthesis
                close_idx = part.find(')')
                if close_idx != -1:
                    score_str = part[:close_idx].strip()
                    try:
                        score = float(score_str)
                        color = self._get_importance_color_simple(score)
                        new_parts.append(f"{color}(importance: {score_str}){ColorCodes.RESET}" + part[close_idx + 1:])
                    except ValueError:
                        new_parts.append("(importance:" + part)
                else:
                    new_parts.append("(importance:" + part)
            message = ''.join(new_parts)

        # Add beautiful icons for different operations
        if "[MEMORY SAVE" in message:
            message = message.replace("[MEMORY SAVE", f"💾 {ColorCodes.MEMORY_SAVE}[MEMORY SAVE")
        if "[MEMORY RETRIEVAL" in message:
            message = message.replace("[MEMORY RETRIEVAL", f"🔍 {ColorCodes.MEMORY_RETRIEVE}[MEMORY RETRIEVAL")
        if "[NEURAL" in message:
            message = message.replace("[NEURAL", f"🧠 {ColorCodes.MEMORY_NEURAL}[NEURAL")

        # Handle batch progress bars with simple string operations
        if "Batches:" in message and "%" in message and "|" in message:
            lines = message.split('\n')
            for i, line in enumerate(lines):
                if "Batches:" in line and "%" in line and "|" in line:
                    # Color the Batches: part
                    line = line.replace("Batches:", f"{ColorCodes.BRIGHT_CYAN}Batches:{ColorCodes.RESET}")
                    # Color percentage
                    parts = line.split('%')
                    if len(parts) >= 2:
                        # Find the percentage number
                        for j in range(len(parts[0]) - 1, -1, -1):
                            if parts[0][j] == ' ':
                                percentage = parts[0][j+1:] + '%'
                                before_percent = parts[0][:j+1]
                                line = before_percent + f"{ColorCodes.BRIGHT_GREEN}{percentage}{ColorCodes.RESET}" + ''.join(parts[1:])
                                break
                    lines[i] = line
            message = '\n'.join(lines)

        return message

    def _get_importance_color_simple(self, score: float) -> str:
        """Get gradient color based on importance score."""
        score_int = int(score * 10)  # Convert 0.1-1.0 to 1-10

        # Find the right color range
        for score_range, color in ColorCodes.IMPORTANCE_GRADIENT.items():
            if score_int in score_range:
                return color + ColorCodes.BOLD

        # Fallback to high importance color
        return ColorCodes.MEMORY_IMPORTANCE_HIGH + ColorCodes.BOLD

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
                # Highlight streaming tokens with simple string operations
                if ":" in message:
                    parts = message.split(":", 1)
                    if len(parts) == 2 and "token" in parts[0].lower():
                        # Color the token part and value separately
                        token_part = f"{ColorCodes.BRIGHT_YELLOW}{parts[0]}:{ColorCodes.RESET}"
                        value_part = parts[1].strip()
                        # Check if value is quoted
                        if (value_part.startswith('"') and value_part.endswith('"')) or \
                           (value_part.startswith("'") and value_part.endswith("'")):
                            value_part = f"{ColorCodes.GREEN}{value_part}{ColorCodes.RESET}"
                        message = f"{token_part} {value_part}"

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
