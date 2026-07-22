# ConvoKB

A lightweight, zero-dependency Python utility to solve "conversation amnesia" in LLM interactions.

ConvoKB allows you to import conversation histories from multiple AI platforms, classify them using an LLM for long-term context, and interact with your data through either a powerful Command Line Interface (CLI) or a beautiful, local Web UI.

## Key Features
- **Zero Dependencies:** 100% pure Python. No `pip install` required.
- **Multi-Platform Import:** Import conversations from ChatGPT, Claude, and Osaurus (or any custom format).
- **AI-Powered Classification:** Uses LLMs (Anthropic, OpenAI-compatible) to automatically summarize, tag, and assess the importance of your conversations.
- **Dual Interfaces:**
 - **CLI:** A robust command-line tool for heavy-duty management, importing, and bulk classification.
 - **Web UI:** A built-in, read-only web server for browsing your conversation timeline, searching by keyword, and filtering by tags, sources, or importance.
- **Privacy-First:** All data is stored locally in a single SQLite database. Your conversations never leave your machine unless you explicitly send them to an LLM for classification.

## Components

### 1. `kb.py` (The Core)
The primary command-line tool for managing your knowledge base.
- **Import:** Ingest JSON/JSONL conversation exports.
- **Search/Filter:** Find specific conversations using keyword search, tags, or metadata flags (like `has_code` or `is_research`).
- **Stats/Tags:** View high-level statistics and trending tags within your database.

### 2. `kb_classify.py` (The Brain)
An automated worker that transforms raw conversation text into structured, searchable knowledge.
- **Backends:** Supports Anthropic, OpenAI-compatible APIs (Osaurus, LM Studio, etc.), and a `mock` backend for testing.
- **Smart Summarization:** Generates concise summaries and assigns importance scores (1-5).
- **Auto-Tagging:** Identifies topics, tags, languages, and structural flags (e.g., `has_images`, `has_code`).

### 3. `kb_web.py` (The Gallery)
A built-in, zero-config web server to visualize your knowledge base.
- **Timeline View:** Browse conversations chronologically.
- **Advanced Filtering:** Use the sidebar to filter by Source, Importance, Tags, or Flags.
- **Full-Text Search:** Rapidly find conversations using keyword matching.

## Installation

Since this project has no external dependencies, you can simply clone the repository.

```bash
git clone https://github.com/quartz-hertz/convo-kb.git
cd ConvoKB
```

## Usage

### Data Ingestion
```bash
# Import conversation files
python3 kb.py import chatgpt.json claude.json
```

### Classification
#### Set ANTHROPIC_API_KEY or OPENAI_API_KEY as appropriate for the --backend method
```bash
# Classify with Anthropic
python3 kb_classify.py --backend anthropic --model claude-3-haiku-20240307

# Classify with a local model (e.g., Osaurus)
python3 kb_classify.py --backend openai --base-url http://127.0.0.1:1337/v1 --model your-model
```

### Browsing
**Start the Web UI:**
```bash
python3 kb_web.py
# Open http://127.0.0.1:8765 in your browser
```

**Using the CLI:**
```bash
# Search for code-heavy research conversations
python3 kb.py search "python architecture" --flag has-code --min-importance 3
```

## Note on Contributions
This project is a personal utility designed to solve a specific problem for my own workflow. While I welcome bug reports and feature suggestions via **GitHub Issues**, I am not currently accepting Pull Requests or actively maintaining this as a community project.

## License
This project is licensed under the MIT License.
