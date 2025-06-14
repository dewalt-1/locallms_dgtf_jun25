#!/usr/bin/env python3
"""
Basic PDF Research Assistant - Workshop Version
Simple script for analyzing a single PDF with local AI
"""

import os
import sys
from pathlib import Path

# Check and install required packages
def install_requirements():
    packages = [
        "langchain",
        "langchain-ollama", 
        "langchain-community",
        "pypdf"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")

install_requirements()

# Check and install required packages
def install_requirements():
    packages = [
        "langchain",
        "langchain-ollama", 
        "langchain-community",
        "pypdf"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")

install_requirements()

# Now import everything
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage

class BasicPDFResearcher:
    def __init__(self, pdf_path):
        print("Setting up PDF Research Assistant...")
        
        # Check if PDF exists
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Setup components
        self.pdf_path = pdf_path
        self.llm = None
        self.model_name = ""
        self.pdf_text = ""
        self.chunks = []
        
        # Load PDF
        self._load_pdf()
        
        # Setup LLM (will ask user to choose)
        self._setup_llm()
        
        print("Ready to research!")

    def _load_pdf(self):
        """Load and process the PDF"""
        print(f"Loading PDF: {Path(self.pdf_path).name}")
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        
        # Combine all text
        self.pdf_text = "\n\n".join([page.page_content for page in pages])
        
        # Split into smaller chunks for the LLM
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Keep chunks small
            chunk_overlap=100,  # Small overlap
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        # Create chunks
        documents = splitter.create_documents([self.pdf_text])
        self.chunks = [doc.page_content for doc in documents]
        
        print(f"Loaded {len(pages)} pages, created {len(self.chunks)} chunks")

    def _setup_llm(self):
        """Setup Ollama LLM with model selection"""
        print("Setting up local LLM...")
        
        # Check if Ollama is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama not responding")
        except:
            print("Ollama is not running!")
            print("Please start Ollama first:")
            print("1. Install Ollama from https://ollama.ai")
            print("2. Run: ollama serve")
            print("3. Pull a model: ollama pull llama2")
            sys.exit(1)
        
        # Get available models
        models_response = requests.get("http://localhost:11434/api/tags")
        available_models = [model['name'] for model in models_response.json().get('models', [])]
        
        if not available_models:
            print("No models found!")
            print("Please pull a model first: ollama pull llama2")
            sys.exit(1)
        
        # Show available models
        print("\nAvailable models:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model}")
        
        # Let user choose
        while True:
            try:
                choice = input(f"\nChoose model (1-{len(available_models)}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_models):
                        selected_model = available_models[idx].split(':')[0]
                        self.model_name = available_models[idx]  # Store full model name
                        break
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nBye!")
                sys.exit(0)
        
        # Setup the LLM
        self.llm = ChatOllama(
            model=selected_model,
            temperature=0.7
        )
        
        print(f"Using model: {self.model_name}")

    def ask(self, question):
        """Ask a question about the PDF"""
        print(f"\n[{self.model_name}] Thinking about: {question}")
        
        # Find most relevant chunks (simple keyword matching)
        relevant_chunks = self._find_relevant_chunks(question, max_chunks=3)
        
        # Create context from relevant chunks
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # Create the prompt
        system_prompt = f"""You are a helpful research assistant. Answer the question based on the provided PDF content.

PDF Content:
{context}

Be specific and cite relevant parts when possible. If the information isn't in the PDF, say so."""

        # Get response from LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def _find_relevant_chunks(self, question, max_chunks=3):
        """Simple keyword-based chunk selection"""
        question_words = set(question.lower().split())
        
        # Score chunks by keyword overlap
        chunk_scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words.intersection(chunk_words))
            chunk_scores.append((chunk, overlap))
        
        # Sort by score and return top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in chunk_scores[:max_chunks] if score > 0]

    def _switch_model(self):
        """Switch to a different model during chat"""
        print(f"\nCurrent model: {self.model_name}")
        
        # Get available models again
        import requests
        models_response = requests.get("http://localhost:11434/api/tags")
        available_models = [model['name'] for model in models_response.json().get('models', [])]
        
        print("\nAvailable models:")
        for i, model in enumerate(available_models, 1):
            marker = " (current)" if model == self.model_name else ""
            print(f"  {i}. {model}{marker}")
        
        while True:
            try:
                choice = input(f"\nChoose model (1-{len(available_models)}) or 'cancel': ").strip()
                
                if choice.lower() == 'cancel':
                    print("Keeping current model")
                    return
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_models):
                        new_model = available_models[idx]
                        if new_model != self.model_name:
                            selected_model = new_model.split(':')[0]
                            self.model_name = new_model
                            self.llm = ChatOllama(model=selected_model, temperature=0.7)
                            print(f"Switched to: {self.model_name}")
                        else:
                            print("Already using that model")
                        return
                
                print("Please enter a valid number or 'cancel'")
            except (KeyboardInterrupt, EOFError):
                print("\nKeeping current model")
                return

    def chat(self):
        """Interactive chat mode"""
        print(f"\nChat mode - Ask questions about: {Path(self.pdf_path).name}")
        print(f"Current model: {self.model_name}")
        print("Type 'quit' to exit, 'model' to switch models\n")
        
        while True:
            try:
                question = input("Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'model':
                    self._switch_model()
                    continue
                
                if not question:
                    continue
                
                answer = self.ask(question)
                print(f"\nAnswer:\n{answer}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

def main():
    if len(sys.argv) != 2:
        print("Basic PDF Research Assistant")
        print("\nUsage:")
        print("  python basic_pdf_research.py <pdf_file>")
        print("\nExample:")
        print("  python basic_pdf_research.py research_paper.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        # Create researcher
        researcher = BasicPDFResearcher(pdf_path)
        
        # Start interactive chat
        researcher.chat()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()