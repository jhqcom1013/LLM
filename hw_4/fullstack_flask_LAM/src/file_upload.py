from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import openai
import numpy as np
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
documents = []
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Initialize Pinecone

pc = Pinecone()
index_name = "lung-disease-lam-recent-study"
index = pc.Index(index_name)
index.describe_index_stats()


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def get_embeddings(split_docs):
    # Generate embeddings for the text chunks
    texts = [doc.page_content for doc in split_docs]
    document_embeddings = embeddings.embed_documents(texts)
    return document_embeddings

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load and process PDF
def process_pdf(pdf_path):
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    return split_docs

# Directory containing PDFs
pdf_dir = "C:/D/Jane/AI_study/LLM/hw_4/papers/"

with open("C:\D\Jane\AI_study\LLM\hw_4\weblinks.txt", encoding='utf-8', errors='ignore') as in_file:
    links = in_file.read().split('\n')

# Process and upload PDFs
documents = []
for idx, pdf_file in enumerate(os.listdir(pdf_dir)):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Processing {pdf_file}...")
        
        # Extract and split text
        split_docs = process_pdf(pdf_path)
       
        # Get embeddings for split documents
        document_embeddings = get_embeddings(split_docs)
        print("finished embedding")
        documents_with_metadata = [
        {"values": document_embeddings[i], 
         "id" : f"{pdf_file}_chunk_{i}", 
         "metadata": {"filename": pdf_file, 
                      "source_url": links[idx],
                        "text": doc.page_content}
        }
         for i, doc in enumerate(split_docs)
        ]
        print(f"Uploading {len(documents_with_metadata)} chunks to Pinecone...")
# used #vector_store.add_documents(documents=document_chunks, ids=uuids) always cause dictionory error. so I switch into upsert
        index.upsert(documents_with_metadata)
    

print(f"PDF data uploaded to Pinecone index: {index_name}")