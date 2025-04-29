#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from pathlib import Path
import json

from haystack import Pipeline
from haystack.dataclasses import Document, ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
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
    def chunk_documents(docs, chunk_size=250, overlap=50):
        chunked = []
        for doc in docs:
            words = doc.content.split()
            if len(words) <= chunk_size:
                chunked.append(doc)
            else:
                for i in range(0, len(words), chunk_size - overlap):
                    end_idx = min(i + chunk_size, len(words))
                    chunk_text = " ".join(words[i : end_idx])
                    meta = doc.meta.copy()
                    meta["chunk"] = i // (chunk_size - overlap)
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
    # Embedding retriever
    embedding_retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=25,
        scale_score=True
    )
    # BM25 retriever for keyword-based search
    bm25_retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=25
    )
    generator = OpenAIGenerator(model="gpt-4-turbo")
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
    
    # Custom hybrid retrieval component
    from haystack.components.joiners.document_joiner import DocumentJoiner
    
    # Create a joiner to combine results from both retrievers
    document_merger = DocumentJoiner(join_mode="concatenate")
    
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("embedding_retriever", embedding_retriever)
    rag_pipeline.add_component("bm25_retriever", bm25_retriever)
    rag_pipeline.add_component("document_merger", document_merger)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", generator)
    
    # Connect the components
    rag_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    rag_pipeline.connect("embedding_retriever.documents", "document_merger.documents")
    rag_pipeline.connect("bm25_retriever.documents", "document_merger.documents")
    rag_pipeline.connect("document_merger.documents", "prompt_builder.documents")
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

def parse_course_file(course_code: str) -> tuple[dict[str, str], Path] | tuple[None, None]:
    """
    Load and split the raw course .txt file into sections by header.
    Returns a dict of section title -> content, and the file path.
    """
    # locate the course text file
    base = Path(__file__).resolve().parent / "knowledge-base-course" / "heinz_courses_txt"
    # normalized file name uses underscore
    fname = f"{course_code}.txt"
    file_path = base / fname
    # fallback to hyphen if needed
    if not file_path.exists():
        alt = f"{course_code.replace('_','-')}.txt"
        file_path = base / alt
        if not file_path.exists():
            return None, None
    # read content
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        text = file_path.read_text(encoding="latin-1")
    lines = text.splitlines()
    sections: dict[str, str] = {}
    current: str | None = None
    for line in lines:
        if line.startswith("# "):
            # title line
            sections['Title'] = line.lstrip('# ').strip()
            current = None
        elif line.startswith("## "):
            # section header
            header = line.lstrip('# ').strip()
            sections[header] = ''
            current = header
        else:
            if current:
                sections[current] += line + '\n'
    return sections, file_path

def get_course_info(course_code: str, query: str) -> tuple[str, Path] | tuple[None, None]:
    """
    Given a course code and user query, return the exact matching section(s) of the course file.
    """
    sections, file_path = parse_course_file(course_code)
    if not sections:
        return None, None
    q = query.lower()
    # map query keywords to section titles
    keyword_map = {
        'prerequisite': 'Prerequisites',
        'learning outcome': 'Learning Outcomes',
        'description': 'Description',
        'course information': 'Course Information',
        'unit': 'Course Information',
        'syllabus': 'Syllabus Links',
    }
    # if user asks for a specific section, return only that
    for key, section in keyword_map.items():
        if key in q and section in sections:
            content = sections[section].strip()
            # format as markdown
            answer = f"**{sections.get('Title','')}**\n\n**{section}:**\n{content}"
            return answer, file_path
    # otherwise return full course info in a standard order
    parts: list[str] = []
    # title
    if 'Title' in sections:
        parts.append(f"**{sections['Title']}**")
    # ordered keys
    for section in ['Course Information', 'Description', 'Learning Outcomes', 'Prerequisites', 'Syllabus Links']:
        if section in sections:
            text = sections[section].strip()
            parts.append(f"**{section}:**\n{text}")
    answer = '\n\n'.join(parts)
    return answer, file_path

def answer_query(document_store, pipeline, query, profile, chat_history=None):
    """Process a query through the RAG pipeline including user profile and chat history"""
    # Check for greetings and respond directly without RAG
    import re
    greeting_pattern = r'\b(hi|hello|hey|greetings|howdy)\b'
    if re.search(greeting_pattern, query.lower()) and len(query.split()) < 4:
        return f"Hello! I'm your CMU course advisor. How can I help you with course selection today? Are you looking for courses in a specific area?", []
    
    # Try to detect a specific course code in the user query
    course_code = extract_course_number(query)
    # If a code was found, try to return exact course info directly
    if course_code:
        exact, path = get_course_info(course_code, query)
        if exact:
            # return the exact section(s) from the course file without generative LLM
            return exact, [str(path)]
    # If no explicit course code in query, attempt to infer by matching course title
    if not course_code:
        inferred = extract_course_title(document_store, query)
        if inferred:
            course_code = inferred
    
    embedding_retriever_params = {}
    bm25_retriever_params = {"query": query}
    
    # First try with exact course filtering if a course code was found
    if course_code:
        embedding_retriever_params["filters"] = {"field": "course", "operator": "==", "value": course_code}
        embedding_retriever_params["top_k"] = 30  # Increased from 15 to get more comprehensive document coverage
        bm25_retriever_params["filters"] = {"field": "course", "operator": "==", "value": course_code}
    
    # Format chat history for context
    history_context = ""
    if chat_history:
        for msg in chat_history[-4:]:  # Use last 4 messages for context
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history_context += f"{role.capitalize()}: {content}\n"
    
    # Run the RAG pipeline with hybrid retrieval
    result = pipeline.run({
        "text_embedder": {"text": query},
        "embedding_retriever": embedding_retriever_params,
        "bm25_retriever": bm25_retriever_params,
        "prompt_builder": {
            "profile": profile, 
            "query": query,
            "chat_history": history_context
        }
    })
    
    answer = result["generator"]["replies"][0]
    
    # Get similar documents for sources attribution from both retrievers
    # First use the embedding retriever results
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

def evaluate_retrievers(document_store, sample_queries, k=5):
    """Evaluate BM25 and Embedding retrievers on sample queries."""
    bm25_retriever = InMemoryBM25Retriever(document_store=document_store)
    emb_retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=k,
        scale_score=True
    )
    results = {}
    for name, retr in [("BM25", bm25_retriever), ("Embedding", emb_retriever)]:
        top1_correct = 0
        topk_correct = 0
        for query, expected in sample_queries:
            expected_meta = expected.replace("-", "_")
            # Use the appropriate retrieval method depending on the retriever implementation
            if hasattr(retr, "retrieve"):
                docs = retr.retrieve(query=query, top_k=k)
            elif hasattr(retr, "get_top_k"):
                docs = retr.get_top_k(query, k)
            else:
                raise AttributeError(f"Retriever {name} has no 'retrieve' or 'get_top_k' method")
            retrieved_codes = [doc.meta.get("course") for doc in docs]
            if retrieved_codes:
                if retrieved_codes[0] == expected_meta:
                    top1_correct += 1
                if expected_meta in retrieved_codes:
                    topk_correct += 1
        total = len(sample_queries)
        results[name] = {
            "top1": top1_correct / total,
            "topk": topk_correct / total
        }
    print("\nRetriever Evaluation:")
    header = f"{'Retriever':<15}{'Top-1 Acc':>12}{'Top-'+str(k)+' Acc':>12}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(f"{name:<15}{metrics['top1']*100:12.2f}{metrics['topk']*100:12.2f}")

if __name__ == "__main__":
    # Smoke-test retrievers
    document_store = build_index()
    sample_queries = [
        ("Which course covers machine learning with Python?", "95-865"),
        ("Intro to econometric theory prerequisites?", "90-906"),
        ("What programming course teaches data structures?", "95-891"),
        ("Which course introduces database systems?", "95-741"),
        ("Which advanced AI course requires linear algebra?", "95-843"),
    ]
    evaluate_retrievers(document_store, sample_queries, k=5)
    # Interactive or test mode
    interactive_mode = True
    if interactive_mode:
        run_chat()
    else:
        run_test_query()