# RAG Evaluation Tool for CMU Course Advisor

This tool evaluates the performance of the CMU Course Advisor RAG system using various retrieval and generation metrics.

## Metrics

The evaluation script computes the following metrics:

- **Precision@K**: What percentage of top-K retrieved results are relevant?
- **Recall@K**: What percentage of all relevant documents were found in top-K results?
- **ROUGE-1 F1**: Word-level overlap between generated answers and expected keywords
- **ROUGE-L F1**: Longest common subsequence-based overlap
- **BLEU**: N-gram precision-based evaluation of generated text

## Usage

```
python evaluate_rag.py [options]
```

### Options

- `--bm25`: Use BM25 retriever instead of default embedding retriever
- `--compare`: Run evaluation on both retrievers and compare results side-by-side
- `--limit N`: Limit evaluation to first N queries (useful for quick testing)
- `--k K`: Set K value for Precision@K and Recall@K metrics (default: 5)
- `--timeout T`: Set timeout in seconds for each query (default: 120)
- `--output FILE`: Save results to specified CSV file

### Examples

Run basic evaluation with embedding retriever:
```
python evaluate_rag.py
```

Run evaluation with BM25 retriever:
```
python evaluate_rag.py --bm25
```

Compare both retrievers with limited queries:
```
python evaluate_rag.py --compare --limit 10
```

Run evaluation with a shorter timeout:
```
python evaluate_rag.py --limit 5 --timeout 30
```

Save comparison results to CSV:
```
python evaluate_rag.py --compare --output results.csv
```

## Test Queries

The evaluation uses a set of test queries with expected answer keywords from the `test_queries_combined.json` file. These queries cover various course information scenarios including:

- Course descriptions
- Prerequisites
- Instructors
- Textbooks
- Learning objectives
- Assignment details

## Interpreting Results

Higher scores indicate better performance:

- **Precision@K and Recall@K**: Range from 0-1, with 1 being perfect
- **ROUGE-1 and ROUGE-L**: Range from 0-1, with higher values indicating better overlap
- **BLEU**: Range from 0-1, with higher values indicating better n-gram precision

When comparing retrievers, the tool provides both absolute and percentage differences to help you understand which approach works better for your specific use case.

## Troubleshooting

If you encounter timeout issues:
1. Use `--limit` to test with fewer queries
2. Use `--timeout` to adjust the timeout duration per query
3. Check OpenAI API usage and limits

## Analysis

In most cases, embedding-based retrieval tends to perform better for semantic search tasks, while BM25 is often stronger for keyword-based searches. Your specific results may vary depending on:

1. The nature of your queries
2. The content of your course documents
3. The quality of your embeddings
4. The specific parameters used for each retriever

Use the comparison feature to determine which retriever best suits your specific needs.