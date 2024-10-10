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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

def process_web_content(url, input_question):
    """
    Function to load documents from a webpage, split them into chunks, create embeddings,
    and use a retriever chain to answer a query based on the content.
    
    Args:
        url (str): The URL of the webpage to load the content from.
        input_question (str): The question to be answered based on the webpage's content.
    
    Returns:
        str: The response from the document retrieval chain.
    """
    # Step 1: Load the web content
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # Step 2: Initialize Language Model (LLM)
    llm = ChatOpenAI()

    # Step 3: Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    
    # Step 4: Create embeddings for the document
    embedding = OpenAIEmbeddings()
    vectorstoredb = FAISS.from_documents(documents, embedding)
    
    # Step 5: Use the vectorstore as a retriever
    retriever = vectorstoredb.as_retriever()
    
    # Step 6: Create a prompt template for the LLM
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        """
    )
    
    # Step 7: Create a document chain that processes the content
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Step 8: Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Step 9: Invoke the retrieval chain with the provided input question
    response = retrieval_chain.invoke({"input": input_question})
    
    # Return the answer from the response
    return response['answer']

# Streamlit app
def main():
    st.title("LangChain Web Content Retriever")

    # User inputs for the URL and question
    url = st.text_input("Enter the URL of the webpage you want to analyze", 
                        "https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
    
    input_question = st.text_input("Enter your question", 
                                   "LangSmith has two usage limits: total traces and extended")
    
    # When the button is clicked, process the user input and provide the result
    if st.button("Get Answer"):
        with st.spinner('Processing...'):
            try:
                # Process the web content and get the answer
                answer = process_web_content(url, input_question)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
