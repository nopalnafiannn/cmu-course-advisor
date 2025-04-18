# CMU Course Advisor Chatbot

An interactive chatbot that helps students find and explore Carnegie Mellon University courses using a profile-aware RAG (Retrieval-Augmented Generation) system.

## Features

- Personalized course recommendations based on user interest and experience level
- Interactive chat interface with Streamlit
- Command-line interface for quick questions
- RAG architecture with Haystack for accurate, contextualized responses

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
2. Ask questions about CMU courses

### Command Line Interface

```bash
python haystack_rag_advisor_profiled.py
```

Follow the prompts to enter your interests and ask questions.

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
- `knowledge-base-course/` - Directory containing course data
- `requirements.txt` - Project dependencies

## License

This project is available under the [MIT License](LICENSE).