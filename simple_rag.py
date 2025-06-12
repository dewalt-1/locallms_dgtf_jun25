#!/usr/bin/env python3
"""
Simple RAG Research Assistant
Local PDF analysis with vector search - no external dependencies
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
        "langchain-chroma",
        "pypdf",
        "chromadb"
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
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class SimpleRAGSystem:
    def __init__(self):
        print("Setting up RAG Research Assistant...")
        
        # RAG components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.model_name = ""
        
        # Storage
        self.vector_db_path = "./rag_vectorstore"
        self.loaded_documents = []
        
        # Setup components
        self._setup_llm()
        self._setup_embeddings()
        
        print("Ready! Load some PDFs to start researching.")

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
            print("Please start Ollama first: ollama serve")
            sys.exit(1)
        
        # Get available models
        models_response = requests.get("http://localhost:11434/api/tags")
        available_models = [model['name'] for model in models_response.json().get('models', [])]
        
        if not available_models:
            print("No models found! Please pull a model: ollama pull llama2")
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

    def _setup_embeddings(self):
        """Setup local embeddings"""
        print("Setting up embeddings...")
        
        # Use Ollama embeddings (requires nomic-embed-text model)
        try:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            # Test the embeddings
            test_embedding = self.embeddings.embed_query("test")
            print("Using Ollama embeddings (nomic-embed-text)")
        except Exception as e:
            print("nomic-embed-text model not found!")
            print("Please install: ollama pull nomic-embed-text")
            
            install_choice = input("\nInstall nomic-embed-text automatically? (y/N): ").strip().lower()
            if install_choice == 'y':
                print("Installing nomic-embed-text...")
                os.system("ollama pull nomic-embed-text")
                try:
                    self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    test_embedding = self.embeddings.embed_query("test")
                    print("Successfully installed and loaded nomic-embed-text")
                except:
                    print("Installation failed. Please install manually: ollama pull nomic-embed-text")
                    sys.exit(1)
            else:
                sys.exit(1)

    def load_pdfs(self, pdf_sources):
        """Load PDFs from various sources"""
        pdf_files = []
        
        # Handle different input types
        if isinstance(pdf_sources, str):
            if os.path.isdir(pdf_sources):
                # Directory - find all PDFs
                pdf_files = glob.glob(os.path.join(pdf_sources, "*.pdf"))
                print(f"Found {len(pdf_files)} PDFs in directory: {pdf_sources}")
            else:
                # Single file or pattern
                pdf_files = glob.glob(pdf_sources)
                print(f"Found {len(pdf_files)} PDFs matching pattern: {pdf_sources}")
        elif isinstance(pdf_sources, list):
            # List of files
            pdf_files = [f for f in pdf_sources if f.endswith('.pdf') and os.path.exists(f)]
            print(f"Processing {len(pdf_files)} PDF files from list")
        
        if not pdf_files:
            print("No PDF files found!")
            return False
        
        # Process PDFs
        all_documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        print("Processing PDFs...")
        for pdf_file in pdf_files:
            try:
                print(f"Loading: {Path(pdf_file).name}")
                
                # Load PDF
                loader = PyPDFLoader(pdf_file)
                pages = loader.load()
                
                # Add metadata
                for page in pages:
                    page.metadata.update({
                        'source': Path(pdf_file).name,
                        'title': Path(pdf_file).stem,
                        'file_path': pdf_file
                    })
                
                # Split into chunks
                chunks = text_splitter.split_documents(pages)
                all_documents.extend(chunks)
                
                # Track loaded documents
                self.loaded_documents.append({
                    'filename': Path(pdf_file).name,
                    'title': Path(pdf_file).stem,
                    'path': pdf_file,
                    'pages': len(pages),
                    'chunks': len(chunks)
                })
                
                print(f"  {len(pages)} pages, {len(chunks)} chunks")
                
            except Exception as e:
                print(f"  Error processing {pdf_file}: {e}")
        
        if not all_documents:
            print("No documents could be processed!")
            return False
        
        print(f"Total: {len(all_documents)} document chunks from {len(self.loaded_documents)} PDFs")
        
        # Create vector store
        self._create_vectorstore(all_documents)
        return True

    def _create_vectorstore(self, documents):
        """Create vector store and RAG chain"""
        print("Creating vector database...")
        
        try:
            # Create or update Chroma vector store
            if os.path.exists(self.vector_db_path):
                print("Loading existing vector database...")
                self.vectorstore = Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embeddings
                )
                # Add new documents
                self.vectorstore.add_documents(documents)
            else:
                print("Creating new vector database...")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.vector_db_path
                )
            
            # Create retriever with MMR (Maximum Marginal Relevance)
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,        # Number of documents to retrieve
                    "fetch_k": 20, # Number of documents to fetch before MMR
                    "lambda_mult": 0.7  # Diversity vs relevance balance
                }
            )
            
            # Create QA chain
            self._create_qa_chain()
            
            print("Vector database ready!")
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False

    def _create_qa_chain(self):
        """Create the RAG QA chain"""
        # Custom prompt for academic research
        prompt_template = """You are a helpful research assistant analyzing academic papers. Use the provided context to answer the question accurately.

Context from research papers:
{context}

Question: {question}

Instructions:
- Provide detailed, well-researched answers based on the context
- Cite specific papers when making claims (mention the source/title)
- If you find conflicting information, acknowledge it
- If you're uncertain about something, say so
- If the answer isn't in the provided context, clearly state that
- Use clear, academic language

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def ask_question(self, question):
        """Ask a question using RAG"""
        if not self.qa_chain:
            return "Please load some PDFs first! Use: load /path/to/pdfs"
        
        print(f"\n[{self.model_name}] Searching knowledge base...")
        
        try:
            # Get answer using RAG
            result = self.qa_chain({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            
            # Show sources used
            if sources:
                source_titles = set()
                for doc in sources:
                    title = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                    source_titles.add(title)
                
                source_list = list(source_titles)[:4]  # Show up to 4 sources
                print(f"Sources: {', '.join(source_list)}")
                if len(source_titles) > 4:
                    print(f"... and {len(source_titles) - 4} more sources")
            
            return answer
            
        except Exception as e:
            return f"Error processing question: {e}"

    def similarity_search(self, question, k=5):
        """Show similar chunks for debugging/exploration"""
        if not self.vectorstore:
            print("No vector store available. Load PDFs first.")
            return
        
        print(f"\nFinding {k} most similar chunks to: '{question}'")
        
        try:
            docs = self.vectorstore.similarity_search(question, k=k)
            
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                print(f"\n{i}. Source: {source}")
                print(f"   Content: {content}")
                
        except Exception as e:
            print(f"Error in similarity search: {e}")

    def list_documents(self):
        """Show loaded documents"""
        if not self.loaded_documents:
            print("No documents loaded yet.")
            return
        
        print(f"\nLoaded Documents ({len(self.loaded_documents)}):")
        total_chunks = 0
        for i, doc in enumerate(self.loaded_documents, 1):
            print(f"  {i}. {doc['filename']} ({doc['pages']} pages, {doc['chunks']} chunks)")
            total_chunks += doc['chunks']
        
        print(f"\nTotal: {total_chunks} searchable chunks")
        
        # Show vector store info
        if self.vectorstore:
            try:
                collection = self.vectorstore._collection
                count = collection.count()
                print(f"Vector database: {count} embeddings stored")
            except:
                print("Vector database: Active")

    def reset_database(self):
        """Clear the vector database"""
        try:
            if os.path.exists(self.vector_db_path):
                import shutil
                shutil.rmtree(self.vector_db_path)
                print("Vector database cleared")
            
            self.vectorstore = None
            self.retriever = None
            self.qa_chain = None
            self.loaded_documents = []
            
        except Exception as e:
            print(f"Error clearing database: {e}")

    def chat(self):
        """Interactive chat mode"""
        print(f"\nRAG Research Assistant")
        print(f"Current model: {self.model_name}")
        print(f"Embeddings: nomic-embed-text")
        
        print("\nCommands:")
        print("  - Ask research questions")
        print("  - 'load <path>' - Load PDFs from directory or pattern")
        print("  - 'list' - Show loaded documents")
        print("  - 'search <query>' - Show similar text chunks")
        print("  - 'reset' - Clear vector database")
        print("  - 'model' - Switch AI models")
        print("  - 'quit' - Exit")
        
        print(f"\nExample: load papers/")
        print(f"Example: What methodologies are commonly used?\n")
        
        while True:
            try:
                user_input = input("Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'list':
                    self.list_documents()
                    continue
                
                if user_input.lower() == 'reset':
                    confirm = input("Clear all loaded documents? (y/N): ").strip().lower()
                    if confirm == 'y':
                        self.reset_database()
                    continue
                
                if user_input.startswith('load '):
                    path = user_input[5:].strip()
                    if path:
                        self.load_pdfs(path)
                    else:
                        print("Please provide a path: load /path/to/pdfs")
                    continue
                
                if user_input.startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        self.similarity_search(query)
                    else:
                        print("Please provide a search query")
                    continue
                
                if user_input.lower() == 'model':
                    self._switch_model()
                    continue
                
                if not user_input:
                    continue
                
                # Ask question using RAG
                answer = self.ask_question(user_input)
                print(f"\nAnswer:\n{answer}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    def _switch_model(self):
        """Switch to a different model"""
        print(f"\nCurrent model: {self.model_name}")
        
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
                    return
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_models):
                        new_model = available_models[idx]
                        if new_model != self.model_name:
                            selected_model = new_model.split(':')[0]
                            self.model_name = new_model
                            self.llm = ChatOllama(model=selected_model, temperature=0.7)
                            # Recreate QA chain with new model
                            if self.qa_chain:
                                self._create_qa_chain()
                            print(f"Switched to: {self.model_name}")
                        return
                
                print("Please enter a valid number or 'cancel'")
            except (KeyboardInterrupt, EOFError):
                return

def main():
    if len(sys.argv) > 1:
        # Command line usage
        pdf_path = sys.argv[1]
        
        try:
            rag = SimpleRAGSystem()
            
            print(f"\nLoading PDFs from: {pdf_path}")
            if rag.load_pdfs(pdf_path):
                rag.chat()
            else:
                print("Failed to load PDFs")
                
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Interactive usage
        print("Simple RAG Research Assistant")
        print("=" * 35)
        
        try:
            rag = SimpleRAGSystem()
            rag.chat()
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()