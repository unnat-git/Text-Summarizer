# Text Summarizer

## Project Overview

A full-stack web application that generates concise summaries from user-provided text. The backend uses a T5-small transformer model served through a FastAPI endpoint, while the frontend provides a minimal interface for text input and summary display.

## Features

- Text input via a web-based interface
- Server-side summarization using a T5-small model with beam search decoding
- Automatic device selection (CUDA, MPS, or CPU)
- Input preprocessing: whitespace normalization, HTML tag removal, case folding
- Asynchronous request handling via FastAPI

## Tech Stack

| Layer      | Technology                         |
|------------|------------------------------------|
| Backend    | Python, FastAPI, Uvicorn           |
| Frontend   | HTML, CSS, JavaScript              |
| ML         | Hugging Face Transformers, PyTorch |
| Model      | T5-small                           |
| Templating | Jinja2                             |

## Model Details

- **Architecture:** T5-small (Text-to-Text Transfer Transformer)
- **Input format:** Raw text prefixed with `summarize:`
- **Tokenization:** `T5Tokenizer` with padding and truncation at 512 tokens
- **Decoding:** Beam search (`num_beams=4`) with `max_length=150` and early stopping
- **Inference:** Runs under `torch.no_grad()` for reduced memory usage

## Project Structure

```
text-summarizer/
├── app/
│   ├── app.py                 # FastAPI application and inference logic
│   ├── static/
│   │   ├── style.css          # Frontend styles
│   │   └── script.js          # Client-side form handling and API calls
│   └── templates/
│       └── index.html         # Main UI template (Jinja2)
├── requirements.txt           # Python dependencies
├── test-cases.md              # Test scenarios and expected outputs
└── .gitignore
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/unnat-git/Text-Summarizer.git
cd Text-Summarizer
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
uvicorn app.app:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## Usage

1. Open the application in a browser.
2. Enter or paste text into the input field.
3. Click **Summarize**.
4. The generated summary appears below the input area.

The `/summarize` endpoint also accepts direct POST requests:

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{ "dialogue": "Your text here." }'
```

## Future Improvements

- Support for additional input formats (PDF, URL extraction)
- Adjustable summary length parameter
- Response caching for repeated inputs
- Containerized deployment (Docker)
- Hosting on a cloud platform (Render, Railway)

---

Built with ❤️ as a portfolio project showcasing NLP, deep learning, and full-stack Python development.
