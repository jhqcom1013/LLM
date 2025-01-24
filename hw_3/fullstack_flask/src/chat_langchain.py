from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from dotenv import load_dotenv
from uuid import uuid4
from .models import db, ChatMessage


load_dotenv()
pc = Pinecone()

print("Connecting to Pinecone index")
index_name = 'medical-models-literature-rag-index'
index = pc.Index(index_name)
index.describe_index_stats()

print("Creating embeddings")
text_field = "text"
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
vectorstore = PineconeVectorStore(index, embeddings, text_field)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

print("Creating chains")
llm = ChatOpenAI()
print("Creating chains")
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True)


def call_chat(question):
    """
    Function to handle the conversational flow and store the chat history.
    """
    answer = ""
    try:
        # Process the conversation and yield tokens
        for chunk in conversation.invoke(question)["answer"]:
            answer += chunk
            yield {"token": chunk}

        # Save chat message to the database
        chat_message = ChatMessage(quituser_id=1, question=question, answer=answer)
        db.session.add(chat_message)
        db.session.commit()

    except Exception as e:
        print(f"Error in call_chat: {e}")
        yield {"error": str(e)}


