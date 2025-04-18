#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pathlib import Path
import json

from haystack import Pipeline
from haystack.dataclasses import Document, ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder

# ─── 1. Load env ───────────────────────────────────────────────────────────────
load_dotenv()  # expects OPENAI_API_KEY in your .env

def build_index():
    """Build and save the document store with course documents"""
    # Use relative paths to the local knowledge-base-course directory
    base_dir = Path(__file__).resolve().parent / "knowledge-base-course"
    # Use text-based course docs and markdown syllabi
    md_paths = [
        base_dir / "heinz_courses_txt",
        base_dir / "syllabi_heinz_courses_md",
    ]
    md_docs = []
    for folder in md_paths:
        # match both .txt (course descriptions) and .md (syllabi) files
        for md_file in folder.rglob("*.*"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    code = md_file.stem.replace("-", "_")
                    doc = Document(
                        content=content,
                        meta={
                            "course": code,
                            "source": str(md_file),
                            "type": "markdown"
                        }
                    )
                    md_docs.append(doc)
            except UnicodeDecodeError:
                # Try with a different encoding if utf-8 fails
                try:
                    with open(md_file, "r", encoding="latin-1") as f:
                        content = f.read()
                        code = md_file.stem.replace("-", "_")
                        doc = Document(
                            content=content,
                            meta={
                                "course": code,
                                "source": str(md_file),
                                "type": "markdown"
                            }
                        )
                        md_docs.append(doc)
                    print(f"Note: Read {md_file} with latin-1 encoding")
                except Exception as e:
                    print(f"Error reading file {md_file}: {str(e)}")
    print(f"✅ Loaded {len(md_docs)} docs (text and markdown)")

    # JSON metadata processing skipped; using only text and markdown sources
    # Chunk docs to avoid OpenAI context length errors
    def chunk_documents(docs, chunk_size=100):
        chunked = []
        for doc in docs:
            words = doc.content.split()
            if len(words) <= chunk_size:
                chunked.append(doc)
            else:
                for i in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[i : i + chunk_size])
                    meta = doc.meta.copy()
                    meta["chunk"] = i // chunk_size
                    chunked.append(Document(content=chunk_text, meta=meta))
        return chunked

    md_docs = chunk_documents(md_docs)
    print(f"✅ Chunked to {len(md_docs)} document chunks")

    document_store = InMemoryDocumentStore()
    # Embed documents with OpenAI, limiting batch size to avoid context length errors
    # Check if we have cached embeddings
    import pickle
    import os
    cache_file = "document_embeddings_cache.pkl"
    
    # Load cached embeddings if available
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        with open(cache_file, "rb") as f:
            embedded_docs = pickle.load(f)
            document_store.write_documents(embedded_docs)
            return document_store
    
    # If no cache exists, create embeddings with increased batch sizes
    document_embedder = OpenAIDocumentEmbedder(
        model="text-embedding-3-small",
        meta_fields_to_embed=["title", "course"],
        batch_size=16  # Increased from 1 to 16
    )
    
    # Process documents in larger batches for better performance
    def process_docs_in_batches(docs, batch_size=32):  # Increased from 10 to 32
        all_embedded_docs = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size}...")
            try:
                # Process medium-sized batches (increased from 1)
                small_batch_size = 8  # Increased from 1 to 8
                batch_embedded_docs = []
                for j in range(0, len(batch), small_batch_size):
                    small_batch = batch[j:j+small_batch_size]
                    try:
                        embedded_docs = document_embedder.run(small_batch)["documents"]
                        batch_embedded_docs.extend(embedded_docs)
                        document_store.write_documents(embedded_docs)
                    except Exception as e:
                        print(f"Error embedding document at index {i+j}: {str(e)}")
                        # Try to process the document with shorter content
                        for doc in small_batch:
                            try:
                                # Reduce content size if too large
                                if len(doc.content) > 4000:
                                    doc.content = doc.content[:4000]
                                single_doc = document_embedder.run([doc])["documents"]
                                batch_embedded_docs.extend(single_doc)
                                document_store.write_documents(single_doc)
                            except Exception as e2:
                                print(f"Failed to embed document after truncation: {str(e2)}")
                
                all_embedded_docs.extend(batch_embedded_docs)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Cache the embeddings to disk for future use
        print(f"Saving embeddings cache to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(all_embedded_docs, f)
            
        return all_embedded_docs
    
    print("Embedding and indexing document chunks...")
    process_docs_in_batches(md_docs)
    
    print(f"✅ Indexed {document_store.count_documents()} documents with embeddings")
    return document_store

def create_rag_pipeline(document_store):
    """Create the RAG pipeline with user profile support"""
    text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=10,
        scale_score=True
    )
    generator = OpenAIGenerator(model="gpt-4o-mini")
    prompt_builder = PromptBuilder(
        template="""
You are a friendly and helpful course advisor. Respond in a positive, friendly, and encouraging tone.

User Profile:
{{ profile }}

Chat History Context:
{{ chat_history }}

Based on the following documents:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}

Answer the question: {{ query }}

IMPORTANT FORMATTING INSTRUCTIONS:
1. Use markdown formatting in your response for better readability
2. Use **bold** for important course information like course codes and titles
3. Use bullet points or numbered lists when listing multiple items
4. Use headings (##) for section titles if your response has multiple sections

Provide a comprehensive and specific answer that directly addresses the question.
If the question asks about a specific course, mention the course code and title.
If the information isn't available in the documents, state that clearly.

If the user message contains a greeting like "hi", "hello", or similar, respond with a warm greeting.

After providing your answer, include a relevant follow-up question to continue the conversation.
"""
    )
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", generator)
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
    return rag_pipeline

def extract_course_number(query):
    import re
    # Enhanced patterns to match more variations of course numbers
    course_patterns = [
        r"\b\d{2}[_\-\s]?\d{3}\b",  # Matches 90-717, 90_717, 90 717, 90717
        r"\b\d{5}\b",               # Matches 90717 (5 digits together)
        r"\b\d{2}-\d{3}\b",         # Explicit hyphen format
        r"\b\d{2}\s+\d{3}\b"        # Space between numbers
    ]
    
    all_matches = []
    for pattern in course_patterns:
        matches = re.findall(pattern, query)
        all_matches.extend(matches)
    
    if all_matches:
        # Normalize to use underscore format
        return all_matches[0].replace('-', '_').replace(' ', '_')
    
    return None

def extract_course_title(document_store, query: str) -> str | None:
    """
    Attempt to identify a course code by matching a course title in the query.
    Scans indexed course description documents and returns the course code if found.
    """
    title_map: dict[str, str] = {}
    try:
        docs = document_store.get_all_documents()
    except Exception:
        return None
    for doc in docs:
        meta = getattr(doc, 'meta', {})
        if meta.get('type') == 'course_desc':
            title = meta.get('title')
            course = meta.get('course')
            if title and course:
                title_map[title.lower()] = course
    if not title_map:
        return None
    q_lower = query.lower()
    # Check for presence of title phrases, prioritizing longer titles
    for title in sorted(title_map.keys(), key=lambda t: len(t), reverse=True):
        if title in q_lower:
            return title_map[title]
    return None

def answer_query(document_store, pipeline, query, profile, chat_history=None):
    """Process a query through the RAG pipeline including user profile and chat history"""
    # Check for greetings and respond directly without RAG
    import re
    greeting_pattern = r'\b(hi|hello|hey|greetings|howdy)\b'
    if re.search(greeting_pattern, query.lower()) and len(query.split()) < 4:
        return f"Hello! I'm your CMU course advisor. How can I help you with course selection today? Are you looking for courses in a specific area?", []
    
    course_code = extract_course_number(query)
    # If no explicit course code in query, attempt to infer by matching course title
    if not course_code:
        inferred = extract_course_title(document_store, query)
        if inferred:
            course_code = inferred
    retriever_params = {}
    
    # First try with exact course filtering if a course code was found
    if course_code:
        retriever_params["filters"] = {"field": "course", "operator": "==", "value": course_code}
        retriever_params["top_k"] = 15  # Increased from 10 to get more relevant documents
    
    # Format chat history for context
    history_context = ""
    if chat_history:
        for msg in chat_history[-4:]:  # Use last 4 messages for context
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history_context += f"{role.capitalize()}: {content}\n"
    
    # Run the RAG pipeline
    result = pipeline.run({
        "text_embedder": {"text": query},
        "retriever": retriever_params,
        "prompt_builder": {
            "profile": profile, 
            "query": query,
            "chat_history": history_context
        }
    })
    
    answer = result["generator"]["replies"][0]
    
    # Get similar documents for sources attribution
    text_embedder = pipeline.get_component("text_embedder")
    query_embedding = text_embedder.run(text=query)["embedding"]
    
    # Try with course filter first
    similar_docs = []
    if course_code:
        similar_docs = document_store.embedding_retrieval(
            query_embedding=query_embedding,
            top_k=10,
            filters={"field": "course", "operator": "==", "value": course_code}
        )
    
    # If no results with course filter or no course specified, get general results
    if not similar_docs:
        similar_docs = document_store.embedding_retrieval(
            query_embedding=query_embedding,
            top_k=10
        )
    
    # Extract unique sources
    sources = []
    for doc in similar_docs:
        src = doc.meta.get("source")
        if src and src not in sources:
            sources.append(src)
    
    # Add course code to the answer if identified
    if course_code and course_code not in answer:
        # Format the course code in a human-readable way (e.g., 90-717 instead of 90_717)
        pretty_course = course_code.replace('_', '-')
        if not any(pretty_course in s for s in sources[:5]):
            # Only mention if not already in sources
            answer = f"Regarding course {pretty_course}: {answer}"
    # Reorder sources: bump primary course document (e.g., '95_891.md') to front if present
    try:
        import os, re
        for idx, src in enumerate(sources):
            name = os.path.basename(src)
            if re.match(r"^\d{2}_\d{3}\.md$", name):
                # move to front
                sources.insert(0, sources.pop(idx))
                break
    except Exception:
        pass
    return answer, sources[:5]

def run_chat():
    """Interactive chat mode with user profiling"""
    print("💬 CMU Course Advisor Chatbot (Personalized)")
    print("Enter 'exit', 'quit', or 'q' to end the session\n")
    interest = input("What is your interest field? ")
    level = input("What is your current level (e.g., beginner, intermediate, expert)? ")
    user_profile = f"Interest field: {interest}. Current level: {level}."
    print(f"\nUser Profile: {user_profile}\n")
    document_store = build_index()
    pipeline = create_rag_pipeline(document_store)
    while True:
        question = input("\nYour question: ")
        if question.lower() in ("exit", "quit", "q"):
            break
        answer, sources = answer_query(document_store, pipeline, question, user_profile)
        print("\nAnswer:", answer)
        print("\nSources:")
        for source in sources:
            print(f"- {source}")
        print("\n" + "-"*40)

def run_test_query():
    """Non-interactive test mode with a sample query and default profile"""
    print("💬 CMU Course Advisor Chatbot (Test Mode)")
    document_store = build_index()
    pipeline = create_rag_pipeline(document_store)
    test_query = "What is course 95-865 about?"
    default_profile = "Interest field: general. Current level: beginner."
    print(f"\nTest user profile: {default_profile}")
    print(f"\nTest question: {test_query}")
    answer, sources = answer_query(document_store, pipeline, test_query, default_profile)
    print("\nAnswer:", answer)
    print("\nSources:")
    for source in sources:
        print(f"- {source}")

if __name__ == "__main__":
    interactive_mode = True
    if interactive_mode:
        run_chat()
    else:
        run_test_query()