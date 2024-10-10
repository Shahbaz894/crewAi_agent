from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
import os
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Function to initialize the retriever
def initialize_retriever(url, chunk_size=1000, chunk_overlap=150, top_k_results=1, doc_content_chars_max=150):
    # Load the web document
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(docs)

    # Create a FAISS vector database from the document chunks using OpenAI embeddings
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())

    # Return the retriever from the vector database
    retriever = vectordb.as_retriever()

    return retriever


# Function to run the retriever tool manually
def run_retriever(url, query, chunk_size=1000, chunk_overlap=150):
    # Step 1: Initialize the retriever using the provided URL
    retriever = initialize_retriever(url, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 2: Use the retriever to search for the query
    result = retriever.get_relevant_documents(query)

    return result


# Initialize the Wikipedia API Wrapper and Tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=150)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Initialize the Arxiv API Wrapper and Tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=150)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Usage example
url = 'https://blogs.nvidia.com/blog/what-is-a-transformer-model/'
query = "What is a transformer model?"

# Run the retriever manually and get the result
result = run_retriever(url, query)

# Combine tools into a list for future tool selection or usage
tools = [arxiv_tool, wiki_tool]
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "Tell me about transformer"})

# Streamlit UI
st.title('AI-powered Assistant')

# Input from user
user_input = st.text_input("Enter your question")

# If user provides input, generate a response
if user_input:
    # Use the agent to generate a response
    response = agent_executor.invoke({"input": user_input})

    # Display the response in Streamlit
    st.write(response)

