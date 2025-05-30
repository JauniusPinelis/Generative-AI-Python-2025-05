# Basic example of using Google AI to generate content
# this is the same as openai completion but using Google AI's API

# Required packages for this demo:
# uv add google-generativeai python-dotenv pydantic rich
# uv add langchain langchain-google-genai langchain-chroma langchain-community langchainhub beautifulsoup4

import os
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print

# LangChain imports for RAG implementation

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()  # Load environment variables from .env file
# Initialize the Google GenAI client

client = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)

# -----------------------------------------

class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]


response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="List a popular cookie recipe, and include the amounts of ingredients.",
    config={
        "response_mime_type": "application/json",
        "response_schema": Recipe,
    },
)
# Use the response as a JSON string.
print(response.text)

# Use instantiated objects.
recipe: Recipe = response.parsed

print(recipe)

# ---------------------------------------

# we can use langchain with Google AI as well

# ==============================================
# RAG Implementation with LangChain & Google AI
# ==============================================

print("\n" + "="*60)
print("RAG IMPLEMENTATION WITH LANGCHAIN & GOOGLE AI")
print("="*60)

# Setup Google AI components for LangChain
print("\n1. Setting up Google AI components...")

# Initialize LangChain Google AI chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GOOGLE_AI_API_KEY"),
    temperature=0.1,
)

# Initialize Google AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_AI_API_KEY"),
)

print("✓ Google AI LLM and embeddings initialized")

# Document Loading and Processing
print("\n2. Loading and processing documents...")

# Load documents from a web source
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
print(f"✓ Loaded {len(docs)} documents")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
print(f"✓ Split into {len(splits)} chunks")

# Create vector store with Google AI embeddings
print("\n3. Creating vector store...")

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings
)
print("✓ Vector store created with Google AI embeddings")

# Setup RAG chain
print("\n4. Setting up RAG retrieval chain...")

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# Load RAG prompt from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")
print("✓ RAG prompt loaded from LangChain Hub")

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✓ RAG chain assembled")

# Test the RAG system
print("\n5. Testing RAG system...")

test_questions = [
    "What is an AI agent?",
    "What are the main components of an AI agent framework?",
    "How do AI agents plan and execute tasks?",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n--- Question {i}: {question} ---")
    try:
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*60)
print("RAG DEMO COMPLETED")
print("="*60)

# Additional RAG functionality examples
print("\n6. Additional RAG Features...")

# Show retrieved context for transparency
def rag_chain_with_context(question):
    """RAG chain that returns both answer and source context."""
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)
    
    # Generate answer
    answer = rag_chain.invoke(question)
    
    return {
        "question": question,
        "answer": answer,
        "context": context,
        "sources": len(retrieved_docs)
    }

# Test with context visibility
demo_question = "What are the challenges in building AI agents?"
result = rag_chain_with_context(demo_question)

print(f"\nQuestion: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources used: {result['sources']} document chunks")
print(f"\nContext preview (first 200 chars):")
print(f"{result['context'][:200]}...")

# Similarity search demo
print("\n7. Direct similarity search demo...")
similar_docs = vectorstore.similarity_search(
    "agent planning", 
    k=3
)

print(f"Found {len(similar_docs)} similar documents for 'agent planning':")
for i, doc in enumerate(similar_docs, 1):
    print(f"\nDocument {i} preview:")
    print(f"{doc.page_content[:150]}...")

print("\n" + "="*60)
print("COMPLETE RAG DEMONSTRATION FINISHED")
print("Features demonstrated:")
print("- Document loading from web sources")
print("- Text chunking and splitting")
print("- Google AI embeddings integration")
print("- Vector store creation with Chroma")
print("- Retrieval chain with LangChain")
print("- Question answering with context")
print("- Source transparency and similarity search")
print("="*60)