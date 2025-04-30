# CMU Course Advisor

An intelligent system that helps students find and explore Carnegie Mellon University Heinz College courses using a profile-aware RAG (Retrieval-Augmented Generation) architecture.

## Team member
- Naufal Nafian
- Pablo Zavala

## Features

- Personalized course recommendations based on user interest and experience level
- Hybrid retrieval system combining semantic search (embeddings) and keyword search (BM25)
- Interactive chat interface with Streamlit
- Command-line interface for quick questions
- RAG architecture with Haystack for accurate, contextualized responses
- Embedding cache for improved performance

## Architecture

- **Document Processing**: Course descriptions and syllabi are indexed from markdown/text files
- **Retrieval System**: Combines OpenAI embeddings (text-embedding-3-small) with BM25 retrieval
- **Generation**: Uses GPT-4 Turbo to create natural language responses
- **User Profiling**: Captures interest fields and experience levels to personalize recommendations
- **Evaluation Framework**: Supports measurement of retrieval and generation quality metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cmu-course-advisor.git
cd cmu-course-advisor

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

This will open a web interface where you can:
1. Enter your interests and experience level
2. Ask questions about CMU Heinz College courses

### Command Line Interface

```bash
python haystack_rag_advisor_profiled.py
```

Follow the prompts to enter your interests and ask questions.

### Evaluation

```bash
python evaluate_rag.py
```

Runs the evaluation framework to measure system performance.

## Troubleshooting

If you encounter issues:

1. Ensure your OpenAI API key is valid and properly set in the `.env` file
2. Verify you're using Python 3.9 or higher
3. Make sure all dependencies are installed correctly
4. For Streamlit performance improvements, install watchdog:
   ```bash
   pip install watchdog
   ```

## Project Structure

- `streamlit_app.py` - Web interface built with Streamlit
- `haystack_rag_advisor_profiled.py` - Core RAG functionality
- `evaluate_rag.py` - Evaluation framework
- `knowledge-base-course/` - Directory containing course data
  - `heinz_courses_md/` - Course descriptions in markdown
  - `heinz_courses_txt/` - Course descriptions in text
  - `syllabi_heinz_courses_md/` - Course syllabi
- `document_embeddings_cache.pkl` - Cached document embeddings
- `requirements.txt` - Project dependencies

## License

This project is available under the [MIT License](LICENSE).