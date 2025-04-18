#!/usr/bin/env python3
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
print(f"OPENAI_API_KEY is set: {bool(os.environ.get('OPENAI_API_KEY'))}")

# Test haystack imports
try:
    from haystack.dataclasses import Document
    from haystack import Pipeline
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.builders import PromptBuilder
    print("✅ All haystack imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")

# Test basic haystack functionality
try:
    document_store = InMemoryDocumentStore()
    doc = Document(content="Test content")
    document_store.write_documents([doc])
    print(f"✅ Document store created with {document_store.count_documents()} document")
except Exception as e:
    print(f"❌ Error creating document store: {e}")

# Create a minimal pipeline
try:
    pipeline = Pipeline()
    pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))
    print("✅ Pipeline created successfully")
except Exception as e:
    print(f"❌ Error creating pipeline: {e}")

print("Test completed")