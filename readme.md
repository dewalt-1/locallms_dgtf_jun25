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

## macOS
```bash
cd ~/Desktop/workshop_files

# Basic PDF Chat (single document)
python3 basic_pdf_research.py papers/paper1.pdf

# Multi-PDF Chat (multiple documents)  
python3 multi_pdf_research.py papers/

# RAG System (smart search)
python3 simple_rag.py papers/
```

## Windows
```bash
cd C:\Users\YourName\Desktop\workshop_files

# Basic PDF Chat (single document)
python basic_pdf_research.py papers/paper1.pdf

# Multi-PDF Chat (multiple documents)
python multi_pdf_research.py papers/

# RAG System (smart search) 
python simple_rag.py papers/
```
