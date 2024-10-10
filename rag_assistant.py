import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import streamlit as st

# Load environment variables
load_dotenv()

# Test if API key is loaded properly
# print(os.getenv("OPENAI_API_KEY"))  # Debugging line

# Set environment variables for OpenAI and Langsmith API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Initialize the OpenAI LLM model
llm = ChatOpenAI(model="gpt-4")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert AI Engineer. Provide me answers based on the questions."),
        ("user", "{input}")
    ]
)

# Streamlit UI
st.title('AI-powered Finance Assistant')

# Input from user
user_input = st.text_input("Enter your question")

# If user provides input, generate a response
if user_input:
    # Use the prompt and llm in a chain to generate a response
    chain = prompt | llm
    response = chain.invoke({"input": user_input})
    
    # Display the response in Streamlit
    st.write(response["content"])
