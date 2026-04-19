# Development Guide

## Development Environment Setup

```bash
git clone https://github.com/Moge800/yolox_wrapper.git
cd yolox_wrapper
uv sync --group dev
```

---

## Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src/yolox_wrapper
```

### Test Structure

| File | Target | Content |
|------|--------|---------|
| `tests/test_config.py` | `config.py` | AppConfig read/write, profile management |
| `tests/test_dataset.py` | `dataset.py` | DatasetPreparer split and remap processing |
| `tests/test_wrapper.py` | `wrapper.py` | Model initialization, letterbox, NMS, inference results |

> `test_wrapper.py` covers testable areas without requiring the YOLOX package itself.

---

## Linter and Formatter

```bash
uv run ruff check .        # Lint check
uv run ruff format .       # Format
```

---

## Project Structure

```
yolox_wrapper/
├── main.py                     # GUI entry point
├── config.ini                  # Application settings
├── pyproject.toml              # Package configuration
├── src/
│   └── yolox_wrapper/
│       ├── __init__.py         # Package exports
│       ├── config.py           # Config management (AppConfig, ProfileParams)
│       ├── dataset.py          # Dataset preparation (DatasetPreparer)
│       ├── wrapper.py          # Main wrapper (YOLOX, YOLOXResult, YOLOXBoxes)
│       ├── _trainer.py         # Internal training engine (_YOLOXTrainer)
│       └── gui/
│           ├── __init__.py
│           ├── app.py          # Main window (App)
│           ├── train_tab.py    # Train tab (TrainTab)
│           ├── infer_tab.py    # Infer tab (InferTab)
│           ├── camera_tab.py   # Camera tab (CameraTab)
│           └── export_tab.py   # Export tab (ExportTab)
├── tests/
│   ├── test_config.py
│   ├── test_dataset.py
│   └── test_wrapper.py
└── docs/                       # Documentation
```

---

## Dependency Structure

```
wrapper.py ──→ _trainer.py ──→ dataset.py
    │
    └──→ dataset.py

config.py (independent)

gui/app.py ──→ gui/*_tab.py ──→ wrapper.py, config.py
```

---

## License

MIT License — See [LICENSE](../../LICENSE) for details.
