#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env
print(f"OPENAI_API_KEY is set: {bool(os.environ.get('OPENAI_API_KEY'))}")

try:
    from haystack import Pipeline
    from haystack.dataclasses import Document
    print("Successfully imported Pipeline and Document")
except ImportError as e:
    print(f"Import error: {e}")

print("Test completed")