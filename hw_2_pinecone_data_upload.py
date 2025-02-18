from pinecone import Pinecone
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
load_dotenv()


pc = Pinecone()
index_name = 'medical-research-paper-rag-index'
index = pc.Index(index_name)
index.describe_index_stats()

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

from langchain_pinecone import PineconeVectorStore
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


from uuid import uuid4

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader('C:/D/Jane/AI_study/context_files/', glob="*.pdf")
documents = loader.load()
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document_chunks = text_splitter.split_documents(documents)

uuids = [str(uuid4()) for _ in range(len(document_chunks))]

vector_store.add_documents(documents=document_chunks, ids=uuids)
