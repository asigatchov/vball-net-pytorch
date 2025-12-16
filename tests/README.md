# Tests for VballNetV1c

This directory contains tests for the VballNetV1c model.

## Running Tests

To run all tests:

```bash
python -m pytest tests/
```

To run specific test files:

```bash
python -m pytest tests/test_model_training.py
python -m pytest tests/test_onnx_export.py
```

To run with verbose output:

```bash
python -m pytest tests/ -v
```

## Test Structure

- `test_model_training.py`: Tests for model compilation and training
- `test_onnx_export.py`: Tests for ONNX export functionality
- `conftest.py`: pytest configuration and fixtures