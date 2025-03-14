# app.py

import streamlit as st
import os
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import ollama
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DOCS_DIRECTORY = "./data"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "enhanced-rag"
PERSIST_DIRECTORY = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5


class DocumentProcessor:
    """Class to handle document processing operations"""
    
    @staticmethod
    def get_loader_for_file(file_path: str) -> Optional:
        """Return appropriate loader based on file extension"""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                # Try PyPDFLoader first as it's generally faster
                return PyPDFLoader(file_path=file_path)
            elif file_extension == '.txt':
                return TextLoader(file_path=file_path)
            elif file_extension in ['.csv', '.tsv']:
                return CSVLoader(file_path=file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return UnstructuredExcelLoader(file_path=file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension} for {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {str(e)}")
            return None

    @staticmethod
    def process_file(file_path: str) -> List[Document]:
        """Process a single file and return document chunks"""
        loader = DocumentProcessor.get_loader_for_file(file_path)
        if not loader:
            return []
        
        try:
            logger.info(f"Loading file: {file_path}")
            data = loader.load()
            logger.info(f"Successfully loaded {len(data)} documents from {file_path}")
            
            # Add source metadata
            for doc in data:
                doc.metadata['source'] = file_path
                doc.metadata['filename'] = Path(file_path).name
            
            return data
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []


class RAGSystem:
    """Main RAG System implementation"""
    
    def __init__(self):
        self.vector_db = None
        self.llm = None
        self.retriever = None
        self.chain = None
    
    def initialize(self):
        """Initialize the RAG components"""
        # Initialize LLM first
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0.1)
        
        # Load or create vector database
        self.vector_db = self._load_vector_db()
        
        if self.vector_db:
            # Create enhanced retriever
            self.retriever = self._create_enhanced_retriever()
            
            # Create the chain
            self.chain = self._create_chain()
            return True
        return False
    
    @staticmethod
    def ingest_documents(docs_directory: str) -> Tuple[List[Document], bool]:
        """Load and process documents from directory using parallel processing"""
        if not os.path.exists(docs_directory):
            logger.error(f"Directory not found: {docs_directory}")
            return [], False
        
        # Get all files in directory
        all_files = []
        for root, _, files in os.walk(docs_directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        if not all_files:
            logger.warning(f"No files found in {docs_directory}")
            return [], False
        
        logger.info(f"Found {len(all_files)} files to process")
        
        # Process files in parallel
        all_docs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(all_files))) as executor:
            future_to_file = {executor.submit(DocumentProcessor.process_file, file): file for file in all_files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    docs = future.result()
                    if docs:
                        all_docs.extend(docs)
                        logger.info(f"Added {len(docs)} documents from {file}")
                except Exception as e:
                    logger.error(f"Error processing {file}: {str(e)}")
        
        if not all_docs:
            logger.error("No documents were successfully processed")
            return [], False
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs, True
    
    @staticmethod
    def split_documents(documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks with optimized parameters"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Documents split into {len(chunks)} chunks")
        return chunks
    
    def _load_vector_db(self) -> Optional[Chroma]:
        """Load or create the vector database"""
        try:
            # Ensure embedding model is available
            ollama.pull(EMBEDDING_MODEL)
            embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
            
            # Check if vector database already exists
            if os.path.exists(PERSIST_DIRECTORY):
                start_time = time.time()
                vector_db = Chroma(
                    embedding_function=embedding,
                    collection_name=VECTOR_STORE_NAME,
                    persist_directory=PERSIST_DIRECTORY,
                )
                logger.info(f"Loaded existing vector database in {time.time() - start_time:.2f} seconds")
                
                # Check if the database is not empty
                if vector_db._collection.count() > 0:
                    return vector_db
                logger.info("Existing vector database is empty, creating new one")
            
            # Load and process documents
            documents, success = RAGSystem.ingest_documents(DOCS_DIRECTORY)
            if not success or not documents:
                return None
            
            # Split documents into chunks
            chunks = RAGSystem.split_documents(documents)
            
            # Create vector database
            start_time = time.time()
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            vector_db.persist()
            logger.info(f"Vector database created and persisted in {time.time() - start_time:.2f} seconds")
            return vector_db
            
        except Exception as e:
            logger.error(f"Error in loading/creating vector database: {str(e)}")
            return None
    
    def _create_enhanced_retriever(self) -> ContextualCompressionRetriever:
        """Create an enhanced retriever with multi-query and compression"""
        try:
            # First create basic retriever
            base_retriever = self.vector_db.as_retriever(
                search_type="mmr",  # Use Maximum Marginal Relevance for diversity
                search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": TOP_K_RETRIEVAL * 2}
            )
            
            # Create multi-query retriever
            query_prompt = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI assistant helping to retrieve relevant document passages.
Generate three different versions of the given question to help find relevant information.
These should be different phrasings or perspectives on the same underlying information need.
Provide these alternative questions separated by newlines.

Original question: {question}"""
            )
            
            multi_query_retriever = MultiQueryRetriever.from_llm(
                base_retriever, self.llm, prompt=query_prompt, 
                include_original=True, search_kwargs={"k": TOP_K_RETRIEVAL}
            )
            
            # Create document compressor for contextual compression
            compressor = LLMChainExtractor.from_llm(self.llm)
            
            # Create contextual compression retriever
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=multi_query_retriever,
            )
            
            logger.info("Enhanced retriever created with multi-query and compression")
            return compression_retriever
        except Exception as e:
            logger.error(f"Error creating enhanced retriever: {str(e)}")
            # Fallback to basic retriever
            return self.vector_db.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    
    def _create_chain(self):
        """Create the RAG chain"""
        # Enhanced RAG prompt
        template = """You are a helpful AI assistant that provides accurate information based on the given context.
Answer the question based ONLY on the following context. If the information isn't in the context, say "I don't have enough information to answer this question." Don't make up answers.

Context:
{context}

Question: {question}

Provide a clear, concise, and informative answer:
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("RAG chain created successfully")
        return chain
    
    def process_query(self, query: str) -> str:
        """Process a user query and return the response"""
        try:
            start_time = time.time()
            response = self.chain.invoke(input=query)
            logger.info(f"Query processed in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"


def main():
    st.set_page_config(
        page_title="Multi-Document RAG Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Multi-Document RAG Assistant")
    st.subheader("Ask questions about your documents")
    
    # Sidebar for settings and info
    with st.sidebar:
        st.header("About")
        st.info(
            "This application uses RAG (Retrieval-Augmented Generation) to answer "
            "questions based on your document library in the data directory."
        )
        
        st.header("Settings")
        if st.button("Rebuild Vector Database"):
            # Remove existing vector store
            import shutil
            if os.path.exists(PERSIST_DIRECTORY):
                shutil.rmtree(PERSIST_DIRECTORY)
                st.success("Vector database cleared. Refresh the page to rebuild.")
        
        st.header("Document Status")
        file_count = 0
        for _, _, files in os.walk(DOCS_DIRECTORY):
            file_count += len(files)
        st.metric("Documents in library", file_count)
    
    # Initialize RAG system
    @st.cache_resource
    def get_rag_system():
        rag_system = RAGSystem()
        success = rag_system.initialize()
        return rag_system, success
    
    rag_system, success = get_rag_system()
    
    if not success:
        st.error("Failed to initialize the RAG system. Please check the logs for details.")
        return
    
    # User input area
    user_input = st.text_area("Enter your question:", height=100)
    
    if st.button("Submit Question", type="primary"):
        if not user_input:
            st.warning("Please enter a question to get started.")
            return
        
        with st.spinner("Searching documents and generating response..."):
            try:
                # Process the query
                response = rag_system.process_query(user_input)
                
                # Display the response
                st.markdown("### Answer")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Show some example questions
    with st.expander("Example questions you can ask"):
        st.markdown("""
        - What are the main topics covered in these documents?
        - Can you summarize the key information about [specific topic]?
        - What are the relationships between [concept A] and [concept B]?
        - When was [specific event] mentioned in the documents?
        - How does [process/system] work according to the documents?
        """)


if __name__ == "__main__":
    main()