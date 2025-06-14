# repository for code for the Local LLMs for Research workshop

## download Ollama:
https://ollama.com

## run Ollama:

```bash
ollama --version
ollama pull llama3.2
ollama run llama3.2
```

# to run our code, cd into the directory and:
```bash
python ollama_pdf.py research_paper.pdf
python ollama_multipdf.py paper1.pdf paper2.pdf 

```

# Simple RAG

## windows
```bash
# 1. Navigate to folder with script
cd ~/Desktop/workshop_files

# 2. Run RAG system
python3 simple_rag.py

# 3. Load your papers (when prompted)
Your question: load papers/
```

## mac
```bash
# 1. Navigate to folder with script
cd ~/Desktop/workshop_files

# 2. Run RAG system
python3 simple_rag.py

# 3. Load your papers (when prompted)
Your question: load papers/
```
