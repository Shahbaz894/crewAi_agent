import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

st.title("Langchain Demo With Gemma Model")
input_text=st.text_input('what questiin you have in mind?')

llm=Ollama(model='gemma:2b')
output_parser=StrOutputParser()
chain=prompt|llm|output_parser
if input_text:
    st.write(chain.invoke({'question':input_text}))