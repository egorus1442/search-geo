import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)

    # add_logger_name/add_log_level требуют логгеры из stdlib logging (с атрибутом .name),
    # поэтому logger_factory должен быть stdlib.LoggerFactory(), а не PrintLoggerFactory().
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level_int)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    return structlog.get_logger(name)
