import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import openai
import streamlit as st
import sentence_transformers
import torch


# Ensure your OpenAI API key is set in your environment
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the API key is set

# Define a simple Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Load and preprocess PDF data
FILE_PATHS = ["migration_development_brief_38_june_2023_0.pdf"]
all_texts = ""

for file_path in FILE_PATHS:
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    texts = [page.page_content for page in pages]  # Extract text from each page
    for text in texts:
        all_texts+= text

# Initialize embeddings using SentenceTransformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_function = SentenceTransformerEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Define a function to process embeddings in batches
def embed_in_batches(texts, embedding_function, batch_size=512):
    """
    Function to process embeddings in smaller batches.

    :param texts: List of text documents to be embedded
    :param embedding_function: Embedding model function
    :param batch_size: Size of each batch (default 512)
    :return: List of embeddings
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = embedding_function.embed_documents(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings

# Create embeddings in batches
embeddings = embed_in_batches(all_texts, embedding_function, batch_size=512)

# Create a vector store using Chroma
vectordb = Chroma(
    collection_name="remittance_pdf",
    embedding_function=embedding_function,
    persist_directory="./vector_db"
)

# Add texts and embeddings to the vector store in batches
vectordb.add_texts(texts=all_texts, embeddings=embeddings)

# Persist vector store
vectordb.persist()


"""# Create a vector store using Chroma
vectordb = Chroma.from_texts(
    texts=all_texts,
    embedding=embedding_function,
    persist_directory="./vector_db",
    collection_name="remittance_pdf"
)



# Persist vector store
vectordb.persist()"""

# Function to retrieve the most relevant chunks from the vector store
def get_relevant_chunks(query, vectordb, top_k=5):
    # Generate embedding for the query
    query_embedding = embedding_function.embed_query(query)
    # Perform a similarity search in the vector store
    results = vectordb.similarity_search_with_score(query_embedding, k=top_k)
    return [result[0].page_content for result in results]


# Function to query OpenAI's API
def query_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use GPT-3.5-turbo or another appropriate model
        messages=[{"role": "system", "content": "You are an assitant that answers questions related to remittance.You break down concepts to make them easy to understand"},
                  {"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.5
    )
    return response.choices[0].message['content'].strip()

# Function to find the most relevant chunk and answer the question
def answer_question(query, vectordb):
    # Retrieve relevant chunks
    relevant_chunks = get_relevant_chunks(query, vectordb)
    # Combine retrieved chunks as context
    context = "\n".join(relevant_chunks)
    # Create a prompt with context and the user query
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    # Query OpenAI's API
    answer = query_openai(prompt)
    return answer

# Streamlit app
st.title("Remittance Chatbot")

st.write("This app allows you to ask questions about remittance using RAG")

prompt = st.chat_input("Ask a question about remittance:")

if prompt:
    st.write(f"Message by user: {prompt}")
    answer = answer_question(prompt, vectordb)
    st.write("Answer:", answer)
else:
    st.write("Please enter a question.")