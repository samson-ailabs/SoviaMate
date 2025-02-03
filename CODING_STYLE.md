# Python Code Style Guide

This document outlines the coding standards for SoviaMate. All contributors must follow these guidelines to maintain consistency and readability.

## 1. General Guidelines
- Use **Python 3.11+** features when available.
- Follow **PEP 8** for code style and formatting.
- Ensure all code is **type-hinted** and properly documented.
- Keep code **modular**, **readable**, and **reusable**.

## 2. Formatting
- Use **4 spaces** for indentation (no tabs).
- Keep lines **≤ 100 characters** (except for URLs and long import statements).
- Use **meaningful variable and function names**.
- Always use **f-strings** (f"Hello {name}") over % formatting or .format().

## 3. Imports
- Use **absolute imports** whenever possible.
- Group imports in the following order:
  1. Standard library imports
  2. Third-party library imports
  3. Local imports
- Example:
  ```python
  import os
  import sys
  
  import numpy as np
  import torch
  
  from soviamate.utils import helper
  ```

## 4. Naming Conventions
- Use **snake_case** for variables and functions:
  ```python
  def process_audio():
      pass
  ```
- Use **PascalCase** for class names:
  ```python
  class AudioProcessor:
      pass
  ```
- Use **UPPER_CASE** for constants:
  ```python
  MAX_BUFFER_SIZE = 4096
  ```

## 5. Type Hinting
- All function arguments and return values must have **type annotations**.
- Use Optional[T] for arguments that can be None.
- Example:
  ```python
  from typing import List, Optional
  def process_text(text: str, max_length: Optional[int] = None) -> List[str]:
      ...
  ```

## 6. Error Handling
- Use exceptions instead of returning error codes.
- Catch specific exceptions and log meaningful messages:
  ```python
  try:
      result = process_audio()
  except ValueError as e:
      logger.error(f"Invalid input: {e}")
  ```

## 7. Testing
- All code must include unit tests using **pytest**.
- Ensure **≥ 90% code coverage**.
- Write test functions in `tests/`.

## 8. Linting & Formatting
- Use **black** for auto-formatting.
- Use **pylint** for linting.
- Use **mypy** for type checking.

## 9. Git Commit Guidelines
- Use clear, descriptive commit messages.
- Follow proper commit message conventions:
  ```
  [Feature] Add real-time speech recognition module
  [Fix] Resolve memory leak in audio processing
  [Refactor] Optimize LLM prompt handling
  ```

## 10. Security & Performance
- Avoid using **eval()** or executing untrusted input.
- Use **async** where necessary for I/O-heavy tasks.
- Profile code using **cProfile** if performance is critical.

---

By following this guide, we ensure code quality, readability, and maintainability. 🚀
