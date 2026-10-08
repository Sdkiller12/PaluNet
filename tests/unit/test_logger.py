import json
import logging

import pytest

from src.utils.logger import PREDICTION_LOGGER, JsonFormatter, configure_logging, get_logger


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    pred = logging.getLogger(PREDICTION_LOGGER)
    saved = (root.level, list(root.handlers), list(pred.handlers))
    yield
    for logger in (root, pred):
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
    root.setLevel(saved[0])
    for h in saved[1]:
        root.addHandler(h)
    for h in saved[2]:
        pred.addHandler(h)


def make_record(msg="bonjour %s", args=("monde",), **extra):
    record = logging.LogRecord("palunet.test", logging.WARNING, __file__, 1, msg, args, None)
    record.__dict__.update(extra)
    return record


def test_json_formatter_includes_message_and_extra_fields():
    payload = json.loads(JsonFormatter().format(make_record(request_id="abc", latency_ms=12.5)))
    assert payload["message"] == "bonjour monde"
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "palunet.test"
    assert payload["request_id"] == "abc" and payload["latency_ms"] == 12.5
    assert payload["timestamp"].endswith("+00:00")
    assert "args" not in payload and "exception" not in payload


def test_json_formatter_serialises_exception():
    try:
        raise RuntimeError("boum")
    except RuntimeError:
        import sys

        record = make_record()
        record.exc_info = sys.exc_info()
    payload = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: boum" in payload["exception"]


def test_configure_logging_sets_level_and_single_json_handler(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    configure_logging()
    configure_logging()
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    configure_logging(level="error")
    assert root.level == logging.ERROR


def test_prediction_log_file_rotates_with_retention(tmp_path):
    log_file = tmp_path / "sub" / "predictions.log"
    configure_logging(prediction_log_file=str(log_file), retention_days=30)
    handlers = logging.getLogger(PREDICTION_LOGGER).handlers
    assert len(handlers) == 1
    fh = handlers[0]
    assert isinstance(fh, logging.handlers.TimedRotatingFileHandler)
    assert fh.backupCount == 30 and fh.when == "MIDNIGHT"

    logging.getLogger(PREDICTION_LOGGER).info("prediction", extra={"label": "Parasitized"})
    fh.flush()
    line = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert line["label"] == "Parasitized"

    configure_logging()
    assert logging.getLogger(PREDICTION_LOGGER).handlers == []


def test_get_logger_configures_root_when_unconfigured():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logger = get_logger("palunet.x")
    assert logger.name == "palunet.x"
    assert len(root.handlers) == 1
