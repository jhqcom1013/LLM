from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from dotenv import load_dotenv
from uuid import uuid4
load_dotenv()
pc = Pinecone()

index_name = 'medical-research-paper-rag-index'
index = pc.Index(index_name)
index.describe_index_stats()

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)

text_field = "text"
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
vectorstore = PineconeVectorStore(index, embeddings, text_field)

# Define whole chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# The function to combine multiple document into one
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Get a prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# for chunk in rag_chain.stream("What is attention?"):
#     print(chunk, end="", flush=True)


print("Creating chains")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True)

# while(True):
#     user_input = input("> ")
#     result = conversation.invoke(user_input)
#     print(result["answer"])

while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break

    # Generate response
    result = conversation.invoke(user_input)
    print(result["answer"])

    # Store chat history in Pinecone
    chat_history = memory.load_memory_variables({})["chat_history"]
    for message in chat_history:
        message_id = str(uuid4())
        vector = embeddings.embed_query(message.content)
        index.upsert([(message_id, vector)])