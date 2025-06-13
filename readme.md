#repository for code for the Local LLMs for Research workshop

##download Ollama:
https://ollama.com

##run Ollama:

'''bash
ollama --version
ollama pull llama3.2
ollama run llama3.2
'''

#to run our code, cd into the directory and:
'''bash
python ollama_pdf.py research_paper.pdf
python ollama_multipdf.py paper1.pdf paper2.pdf 

# Interactive mode
python simple_rag.py

# Load directory immediately
python simple_rag.py papers/

# Load pattern immediately  
python simple_rag.py *.pdf
'''

