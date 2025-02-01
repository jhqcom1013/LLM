
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
from langchain.prompts import PromptTemplate

import time
import datetime

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)


load_dotenv()
pc = Pinecone()

print("Connecting to Pinecone index")
index_name = 'lung-disease-lam-recent-study'
index = pc.Index(index_name)
index.describe_index_stats()

print("Creating embeddings")
text_field = "text"
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
vectorstore = PineconeVectorStore(index, embeddings, text_field)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
custormized_prompt = "Today data is {date}. You are a friendly and helpful chatbot having a conversation with a human.\
Your first priority is to retrieve relevant information from the retriever. If the retriever does not provide an answer or\
the information is insufficient, then generate a response using your own knowledge and reasoning.\
Keep the conversation engaging, clear, and helpful..".format(date=now)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(custormized_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(memory_key="chat_history", 
                                  output_key="answer",
                                  return_messages=True)

print("Creating chains")
llm = ChatOpenAI(model="gpt-4", temperature=0)

def custom_retriever(question):
    """Retrieve documents with similarity scores"""
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(question, k=8)
    
    # Extract documents and their similarity scores
    #documents = [doc[0] for doc in retrieved_docs_with_scores]  # Get document content
    scores = [doc[1] for doc in retrieved_docs_with_scores]  # Get similarity scores
    return scores 

conversation = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, 
    memory=memory, 
    verbose=True, 
    condense_question_prompt=prompt,
    return_source_documents=True,
    get_chat_history=lambda h : h)

def chatbot_response(user_query):
    # Manually retrieve documents & similarity scores
    scores = custom_retriever(user_query)
    print(scores)

    if max(scores) >= 0.86:  # Adjust threshold as needed
        response = conversation.invoke({"question": user_query})
        source_documents = response["source_documents"]  # Includes metadata
        url = list(set([doc.metadata['source_url'] for doc in source_documents]))
        answer = response["answer"]

    else:
        # If similarity is low, rely on ChatGPT alone
        print("Using ChatGPT alone...")
        answer = llm.predict(f"User: {user_query}\nChatbot: ")
        url = "Using ChatGPT alone.."
    
    return answer, url


def call_chat(question):
    """
    Function to handle the conversational flow and store the chat history.
    """
    answer = ""

    answer, url = chatbot_response(question)
   
    # Extract answer and sources
    full_response = answer + f"\n\nSource: {url}"
    print(full_response)

    # Process the conversation and yield tokens
    for chunk in full_response.split("\n"):
        answer += chunk
        yield {"token": chunk}

    # Save chat message to the database
    chat_message = ChatMessage(user_id=1, question=question, answer=answer)
    db.session.add(chat_message)
    db.session.commit()
