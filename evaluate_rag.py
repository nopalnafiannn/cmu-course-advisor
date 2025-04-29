#!/usr/bin/env python3
"""
Evaluation script for the Haystack-based RAG course advising system.
Computes retrieval and generation metrics on a suite of test queries.
Metrics:
  - Precision@K, Recall@K (keyword presence in generated answers)
  - ROUGE-1 F1, ROUGE-L F1
  - BLEU score

Usage:
  python evaluate_rag.py [--bm25] [--compare] [--limit N] [--k K]
  
  --bm25    : Use BM25 retriever instead of embedding retriever
  --compare : Run evaluation on both retrievers and compare results
  --limit N : Limit evaluation to first N queries
  --k K     : Set K value for Precision@K and Recall@K metrics (default: 5)
"""
import os
import sys
import json
import argparse
import signal
import time
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure project root is importable
_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _HERE)

import haystack_rag_advisor_profiled as rag_module
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Query timeout")

def evaluate_pipeline(pipeline, document_store, test_items, K, is_bm25=False, query_timeout=120):
    """Run evaluation on a pipeline with given test items and parameters"""
    # Prepare evaluators
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1  # Apply smoothing to avoid zero BLEU scores

    # Accumulators for metrics
    precision_vals = []
    recall_vals = []
    rouge1_vals = []
    rougeL_vals = []
    bleu_vals = []

    # Set up timeout handling
    signal.signal(signal.SIGALRM, timeout_handler)
    
    print(f'Evaluating {len(test_items)} queries with {"BM25" if is_bm25 else "Embedding"} retriever...')
    for item in tqdm(test_items, desc='Queries'):
        query = item.get('query', '').strip()
        expected = item.get('expected_answer_keywords', []) or []
        # Skip if no expected keywords
        if not expected:
            continue

        try:
            # Set alarm
            signal.alarm(query_timeout)
            
            # Run RAG query
            if is_bm25:
                # For BM25 we need a different query mechanism since we removed text_embedder
                retriever_params = {}
                # Check for course code to filter
                course_code = rag_module.extract_course_number(query)
                if course_code:
                    retriever_params["filters"] = {"field": "course", "operator": "==", "value": course_code}
                    retriever_params["top_k"] = 15
                    
                result = pipeline.run({
                    "retriever": {"query": query, **retriever_params},
                    "prompt_builder": {
                        "profile": "Interest field: general. Current level: beginner.", 
                        "query": query,
                        "chat_history": ""
                    }
                })
                answer = result["generator"]["replies"][0]
            else:
                # Use the standard answer_query function
                answer, _ = rag_module.answer_query(
                    document_store, pipeline, query,
                    profile="Interest field: general. Current level: beginner.", 
                    chat_history=[]
                )
            
            # Cancel alarm
            signal.alarm(0)
            
            ans_lower = answer.lower()

            # Retrieval evaluation: keyword presence
            found = sum(1 for kw in expected if kw.lower() in ans_lower)
            prec = min(found, K) / float(K)
            rec = found / float(len(expected))
            precision_vals.append(prec)
            recall_vals.append(rec)

            # Generation evaluation: ROUGE and BLEU
            reference = ' '.join(expected)
            scores = scorer.score(reference, answer)
            rouge1_vals.append(scores['rouge1'].fmeasure)
            rougeL_vals.append(scores['rougeL'].fmeasure)
            ref_tokens = reference.lower().split()
            hyp_tokens = ans_lower.split()
            try:
                # Use smoothing to handle corner cases
                bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
            except Exception as e:
                print(f"BLEU error: {e}")
                bleu = 0.0
            bleu_vals.append(bleu)
            
        except TimeoutError:
            print(f"\nWarning: Query timed out after {query_timeout} seconds: {query[:50]}...")
            continue
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            continue
        finally:
            # Always cancel alarm to be safe
            signal.alarm(0)

    # Final aggregation
    n = len(precision_vals)
    if n == 0:
        print('No valid test items with expected keywords found.')
        return None
        
    results = {
        "precision": sum(precision_vals) / n,
        "recall": sum(recall_vals) / n,
        "rouge1": sum(rouge1_vals) / n,
        "rougeL": sum(rougeL_vals) / n,
        "bleu": sum(bleu_vals) / n
    }
    
    # Reporting
    retriever_type = "BM25" if is_bm25 else "Embedding"
    print(f'\n===== Evaluation Results ({retriever_type} Retriever) =====')
    print(f'Average Precision@{K}: {results["precision"]:.4f}')
    print(f'Average Recall@{K}:    {results["recall"]:.4f}')
    print(f'Average ROUGE-1 F1:      {results["rouge1"]:.4f}')
    print(f'Average ROUGE-L F1:      {results["rougeL"]:.4f}')
    print(f'Average BLEU:            {results["bleu"]:.4f}')
    
    return results

def create_bm25_pipeline(document_store):
    """Create a new pipeline with BM25 retriever"""
    pipeline = rag_module.Pipeline()
    
    bm25_retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=10
    )
    prompt_builder = rag_module.PromptBuilder(
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
    generator = rag_module.OpenAIGenerator(model="gpt-4o-mini")
    
    pipeline.add_component("retriever", bm25_retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)
    
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")
    
    return pipeline

def print_comparison(embedding_results, bm25_results, K):
    """Print side-by-side comparison of two retriever results"""
    print("\n" + "=" * 80)
    print(f"COMPARISON: Embedding vs BM25 Retrievers (K={K})")
    print("=" * 80)
    
    print(f"{'Metric':<18} {'Embedding':>12} {'BM25':>12} {'Diff (Emb-BM25)':>20} {'% Change':>10}")
    print("-" * 80)
    
    for metric in ["precision", "recall", "rouge1", "rougeL", "bleu"]:
        emb_val = embedding_results[metric]
        bm25_val = bm25_results[metric]
        diff = emb_val - bm25_val
        
        # Calculate percentage change if bm25_val is not zero
        if bm25_val != 0:
            percent = (diff / bm25_val) * 100
            percent_str = f"{percent:+.1f}%"
        else:
            percent_str = "N/A"
            
        # Format the name based on metric
        if metric == "precision":
            name = f"Precision@{K}"
        elif metric == "recall":
            name = f"Recall@{K}"
        elif metric == "rouge1":
            name = "ROUGE-1 F1"
        elif metric == "rougeL":
            name = "ROUGE-L F1"
        else:
            name = "BLEU"
            
        print(f"{name:<18} {emb_val:>12.4f} {bm25_val:>12.4f} {diff:>+20.4f} {percent_str:>10}")
        
    print("=" * 80)
    print("Positive difference indicates embedding retriever performed better.")
    print("=" * 80)
    
    # Add a conclusion summary
    if embedding_results["precision"] > bm25_results["precision"] and embedding_results["recall"] > bm25_results["recall"]:
        winner = "Embedding"
        margin = "significantly" if (embedding_results["precision"]/max(0.0001, bm25_results["precision"])) > 1.5 else "slightly"
    elif bm25_results["precision"] > embedding_results["precision"] and bm25_results["recall"] > embedding_results["recall"]:
        winner = "BM25"
        margin = "significantly" if (bm25_results["precision"]/max(0.0001, embedding_results["precision"])) > 1.5 else "slightly"
    else:
        winner = "Mixed results"
        margin = ""
    
    if winner != "Mixed results":
        print(f"CONCLUSION: {winner} retriever performs {margin} better overall.")
    else:
        print("CONCLUSION: Results are mixed. Consider your specific use case needs.")
    print("=" * 80)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument('--bm25', action='store_true', help='Use BM25 retriever instead of default embedding retriever')
    parser.add_argument('--compare', action='store_true', help='Run evaluation on both retrievers and compare results')
    parser.add_argument('--k', type=int, default=5, help='K value for Precision@K and Recall@K')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of queries to evaluate')
    parser.add_argument('--output', type=str, help='Save results to specified CSV file')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout in seconds for each query')
    args = parser.parse_args()
    
    # Evaluation parameter
    K = args.k

    # Load test queries
    queries_path = os.path.join(_HERE, 'test_queries_combined.json')
    if not os.path.exists(queries_path):
        print(f"Test queries file not found: {queries_path}")
        sys.exit(1)
    with open(queries_path, 'r', encoding='utf-8') as f:
        test_items = json.load(f)
        
    if args.limit:
        test_items = test_items[:args.limit]

    # Initialize RAG components
    print('Building document store and RAG pipeline...')
    document_store = rag_module.build_index()
    
    # Handle comparison mode
    if args.compare:
        # Evaluate embedding retriever
        embedding_pipeline = rag_module.create_rag_pipeline(document_store)
        embedding_results = evaluate_pipeline(embedding_pipeline, document_store, test_items, K, is_bm25=False, query_timeout=args.timeout)
        
        # Evaluate BM25 retriever
        print("\nSwitching to BM25 retriever...")
        bm25_pipeline = create_bm25_pipeline(document_store)
        bm25_results = evaluate_pipeline(bm25_pipeline, document_store, test_items, K, is_bm25=True, query_timeout=args.timeout)
        
        # Print comparison
        print_comparison(embedding_results, bm25_results, K)
        
        # Save to CSV if output file specified
        if args.output:
            import csv
            import datetime
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'query_count', 'k_value', 'metric', 'embedding', 'bm25', 'difference', 'percent_change']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write metrics
                for metric in ["precision", "recall", "rouge1", "rougeL", "bleu"]:
                    emb_val = embedding_results[metric]
                    bm25_val = bm25_results[metric]
                    diff = emb_val - bm25_val
                    
                    # Calculate percent change
                    if bm25_val != 0:
                        percent = (diff / bm25_val) * 100
                    else:
                        percent = 0
                        
                    # Write row
                    writer.writerow({
                        'timestamp': timestamp,
                        'query_count': len(test_items),
                        'k_value': K,
                        'metric': metric,
                        'embedding': f"{emb_val:.6f}",
                        'bm25': f"{bm25_val:.6f}",
                        'difference': f"{diff:.6f}",
                        'percent_change': f"{percent:.2f}"
                    })
                    
            print(f"\nResults saved to {args.output}")
    elif args.bm25:
        # Only BM25 evaluation
        print("Using BM25 retriever...")
        bm25_pipeline = create_bm25_pipeline(document_store)
        results = evaluate_pipeline(bm25_pipeline, document_store, test_items, K, is_bm25=True, query_timeout=args.timeout)
        
        # Save to CSV if output file specified
        if args.output and results:
            import csv
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'query_count', 'k_value', 'retriever', 'metric', 'value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write metrics
                for metric, value in results.items():
                    writer.writerow({
                        'timestamp': timestamp,
                        'query_count': len(test_items),
                        'k_value': K,
                        'retriever': 'bm25',
                        'metric': metric,
                        'value': f"{value:.6f}"
                    })
                    
            print(f"\nResults saved to {args.output}")
    else:
        # Default embedding evaluation
        embedding_pipeline = rag_module.create_rag_pipeline(document_store)
        results = evaluate_pipeline(embedding_pipeline, document_store, test_items, K, is_bm25=False, query_timeout=args.timeout)
        
        # Save to CSV if output file specified
        if args.output and results:
            import csv
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'query_count', 'k_value', 'retriever', 'metric', 'value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write metrics
                for metric, value in results.items():
                    writer.writerow({
                        'timestamp': timestamp,
                        'query_count': len(test_items),
                        'k_value': K,
                        'retriever': 'embedding',
                        'metric': metric,
                        'value': f"{value:.6f}"
                    })
                    
            print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()