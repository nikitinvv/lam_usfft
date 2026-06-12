"""Colored stdout logger with optional file tee, ported from
holotomocupy_mpi/logger_config.py (minus the MPI-rank machinery — lam_usfft
is single-rank, so records carry no rank tag).

Usage:

    from lam_usfft.logger_config import logger, set_log_level, add_file_handler

    set_log_level("INFO")
    add_file_handler("/path/to/run.log")    # optional — tees to plain-text file
    logger.info("Starting reconstruction ...")
    logger.warning("Truncated to 128 angles")
    logger.error("File not found")
"""

import sys
import logging


RESET = "\033[0m"
COLORS = {
    "DEBUG":    "\033[35m",       # magenta
    "INFO":     "\033[32m",       # green
    "WARNING":  "\033[33m",       # yellow
    "ERROR":    "\033[31m",       # red
    "CRITICAL": "\033[1;31m",     # bold red
}


class ColorMessageFormatter(logging.Formatter):
    """Wraps the message text in an ANSI colour code based on its level."""

    def format(self, record):
        color = COLORS.get(record.levelname, "")
        record.msg_colored = f"{color}{record.getMessage()}{RESET}"
        return super().format(record)


def add_file_handler(path):
    """Tee logger output to a plain-text file (no ANSI colours). Idempotent —
    calling it twice with the same path is a no-op."""
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path:
            return h
    fh = logging.FileHandler(path, mode="w")
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return fh


_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(ColorMessageFormatter(
    fmt="%(asctime)s %(msg_colored)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logger = logging.getLogger("lam_usfft")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(_stdout_handler)
logger.propagate = False


def set_log_level(level):
    """Set the lam_usfft logger level from a string (DEBUG/INFO/WARNING/ERROR/CRITICAL)."""
    logger.setLevel(level.upper() if isinstance(level, str) else level)
