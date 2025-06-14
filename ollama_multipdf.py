#!/usr/bin/env python3
"""
Multi-PDF Research Assistant - Workshop Version
Advanced script for analyzing multiple PDFs - perfect for literature reviews!
"""

import os
import sys
from pathlib import Path
import glob

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

class MultiPDFResearcher:
    def __init__(self, pdf_sources):
        print("Setting up Multi-PDF Research Assistant...")
        
        # Setup components
        self.pdf_sources = pdf_sources
        self.llm = None
        self.model_name = ""
        self.documents = {}  # Store documents by filename
        self.all_chunks = []  # All chunks with source info
        
        # Load PDFs
        self._load_pdfs()
        
        # Setup LLM (will ask user to choose)
        self._setup_llm()
        
        print("Ready to research across multiple documents!")

    def _load_pdfs(self):
        """Load and process multiple PDFs"""
        pdf_files = []
        
        # Handle different input types
        if isinstance(self.pdf_sources, str):
            if os.path.isdir(self.pdf_sources):
                # Directory - find all PDFs
                pdf_files = glob.glob(os.path.join(self.pdf_sources, "*.pdf"))
                print(f"Found {len(pdf_files)} PDFs in directory: {self.pdf_sources}")
            else:
                # Single file or pattern
                pdf_files = glob.glob(self.pdf_sources)
                print(f"Found {len(pdf_files)} PDFs matching pattern: {self.pdf_sources}")
        elif isinstance(self.pdf_sources, list):
            # List of files
            pdf_files = [f for f in self.pdf_sources if f.endswith('.pdf') and os.path.exists(f)]
            print(f"Processing {len(pdf_files)} PDF files from list")
        
        if not pdf_files:
            raise FileNotFoundError("No PDF files found!")
        
        # Load each PDF
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        for pdf_file in pdf_files:
            try:
                print(f"Loading: {Path(pdf_file).name}")
                
                # Load PDF
                loader = PyPDFLoader(pdf_file)
                pages = loader.load()
                
                # Store document info
                filename = Path(pdf_file).name
                self.documents[filename] = {
                    'path': pdf_file,
                    'pages': len(pages),
                    'content': "\n\n".join([page.page_content for page in pages])
                }
                
                # Create chunks with source information
                documents = splitter.create_documents([self.documents[filename]['content']])
                for doc in documents:
                    self.all_chunks.append({
                        'content': doc.page_content,
                        'source': filename,
                        'path': pdf_file
                    })
                
                print(f"  {len(pages)} pages, {len(documents)} chunks")
                
            except Exception as e:
                print(f"  Error loading {pdf_file}: {e}")
        
        print(f"Total: {len(self.documents)} documents, {len(self.all_chunks)} chunks")

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
                        self.model_name = available_models[idx]
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

    def list_documents(self):
        """Show all loaded documents"""
        print(f"\nLoaded Documents ({len(self.documents)}):")
        for i, (filename, info) in enumerate(self.documents.items(), 1):
            print(f"  {i}. {filename} ({info['pages']} pages)")

    def ask(self, question, max_sources=3):
        """Ask a question across all PDFs"""
        print(f"\n[{self.model_name}] Searching across {len(self.documents)} documents...")
        
        # Find most relevant chunks from all documents
        relevant_chunks = self._find_relevant_chunks(question, max_chunks=max_sources)
        
        if not relevant_chunks:
            return "No relevant information found in the documents."
        
        # Create context with source attribution
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"Source: {chunk['source']}\nContent: {chunk['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create the prompt
        system_prompt = f"""You are a helpful research assistant analyzing multiple academic documents. Answer the question based on the provided content from various PDFs.

Content from PDFs:
{context}

Instructions:
- Be specific and mention which document(s) you're referencing
- If information comes from multiple sources, compare and contrast
- If the information isn't in the PDFs, say so
- Cite the source document when making claims"""

        # Get response from LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        response = self.llm.invoke(messages)
        
        # Show which documents were used
        sources_used = list(set([chunk['source'] for chunk in relevant_chunks]))
        print(f"Sources consulted: {', '.join(sources_used)}")
        
        return response.content

    def ask_specific_document(self, question, document_name):
        """Ask a question about a specific document"""
        if document_name not in self.documents:
            available = list(self.documents.keys())
            return f"Document '{document_name}' not found. Available: {', '.join(available)}"
        
        print(f"\n[{self.model_name}] Analyzing: {document_name}")
        
        # Find relevant chunks only from this document
        doc_chunks = [chunk for chunk in self.all_chunks if chunk['source'] == document_name]
        relevant_chunks = self._find_relevant_chunks_in_list(question, doc_chunks, max_chunks=3)
        
        if not relevant_chunks:
            return f"No relevant information found in {document_name}."
        
        # Create context
        context = "\n\n---\n\n".join([chunk['content'] for chunk in relevant_chunks])
        
        # Create the prompt
        system_prompt = f"""You are analyzing the document: {document_name}

Document content:
{context}

Answer the question based only on this document. Be specific and cite relevant parts."""

        # Get response from LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def compare_documents(self, question):
        """Compare how different documents address a question"""
        print(f"\n[{self.model_name}] Comparing across documents...")
        
        # Get relevant chunks from each document
        doc_responses = {}
        for doc_name in self.documents.keys():
            doc_chunks = [chunk for chunk in self.all_chunks if chunk['source'] == doc_name]
            relevant_chunks = self._find_relevant_chunks_in_list(question, doc_chunks, max_chunks=2)
            
            if relevant_chunks:
                context = "\n".join([chunk['content'] for chunk in relevant_chunks])
                doc_responses[doc_name] = context[:500] + "..." if len(context) > 500 else context
        
        if not doc_responses:
            return "No relevant information found across documents."
        
        # Create comparison prompt
        comparison_text = "\n\n".join([f"Document: {doc}\nContent: {content}" for doc, content in doc_responses.items()])
        
        system_prompt = f"""Compare how different documents address this question: {question}

Documents:
{comparison_text}

Provide a comparison that:
1. Summarizes each document's perspective
2. Identifies similarities and differences
3. Notes any conflicting information
4. Highlights unique insights from each source"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def _find_relevant_chunks(self, question, max_chunks=3):
        """Find relevant chunks across all documents"""
        return self._find_relevant_chunks_in_list(question, self.all_chunks, max_chunks)

    def _find_relevant_chunks_in_list(self, question, chunk_list, max_chunks=3):
        """Find relevant chunks in a specific list"""
        question_words = set(question.lower().split())
        
        # Score chunks by keyword overlap
        chunk_scores = []
        for chunk in chunk_list:
            chunk_words = set(chunk['content'].lower().split())
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
        print(f"\nMulti-PDF Chat Mode")
        print(f"Current model: {self.model_name}")
        print(f"Documents loaded: {len(self.documents)}")
        
        print("\nCommands:")
        print("  - Ask questions across all documents")
        print("  - 'list' - show all documents")
        print("  - 'doc:filename question' - ask about specific document")
        print("  - 'compare: question' - compare across documents")
        print("  - 'model' - switch models")
        print("  - 'quit' - exit\n")
        
        while True:
            try:
                user_input = input("Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'list':
                    self.list_documents()
                    continue
                
                if user_input.lower() == 'model':
                    self._switch_model()
                    continue
                
                if user_input.startswith('doc:'):
                    # Specific document query
                    parts = user_input[4:].split(' ', 1)
                    if len(parts) == 2:
                        doc_name, question = parts
                        answer = self.ask_specific_document(question, doc_name.strip())
                    else:
                        answer = "Format: doc:filename question"
                
                elif user_input.startswith('compare:'):
                    # Comparison query
                    question = user_input[8:].strip()
                    answer = self.compare_documents(question)
                
                elif not user_input:
                    continue
                
                else:
                    # General query across all documents
                    answer = self.ask(user_input)
                
                print(f"\nAnswer:\n{answer}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

def main():
    if len(sys.argv) < 2:
        print("Multi-PDF Research Assistant")
        print("\nUsage options:")
        print("  python multi_pdf_research.py <directory>           # All PDFs in directory")
        print("  python multi_pdf_research.py *.pdf                # All PDFs matching pattern")
        print("  python multi_pdf_research.py file1.pdf file2.pdf  # Specific files")
        print("\nExamples:")
        print("  python multi_pdf_research.py papers/")
        print("  python multi_pdf_research.py *.pdf")
        print("  python multi_pdf_research.py paper1.pdf paper2.pdf paper3.pdf")
        sys.exit(1)
    
    # Handle different input formats
    if len(sys.argv) == 2:
        # Single argument - could be directory or pattern
        pdf_sources = sys.argv[1]
    else:
        # Multiple arguments - list of files
        pdf_sources = sys.argv[1:]
    
    try:
        # Create researcher
        researcher = MultiPDFResearcher(pdf_sources)
        
        # Start interactive chat
        researcher.chat()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()