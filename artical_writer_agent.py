import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub



# streamlit run artical_writer_agent.py



load_dotenv()

# Access the token from environment variables
token = os.getenv('HF_TOKEN')
if token is None:
    raise ValueError("HF_TOKEN not found in environment variables.")
else:
    print("Token retrieved successfully")

# Hugging Face API setup
# llm = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     huggingfacehub_api_token=api_key,
#     task="text-generation",
# )
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,  # Set temperature explicitly
    max_length=1028,   # Set max_length explicitly
    huggingface_token=token,
    task='text-generation'# Set token explicitly
    
)
# Planner Agent
planner = Agent(
    role='Article Planner',
    goal='Plan a well-structured and informative article on the topic: {topic}.',
    backstory=(
        'You are responsible for planning a comprehensive Medium article about {topic}. Your goal is to create a '
        'detailed outline that covers the topic introduction, in-depth exploration with structured headings, a '
        'mathematical derivation section, real-life applications, pros and cons, and improvement strategies for '
        'model efficiency.'
    ),
    allow_delegation=False,
    verbose=True
)

# Writer Agent
writer = Agent(
    role='Article Writer',
    goal='Write a detailed article with explanations and step-by-step instructions for topic: {topic}.',
    backstory=(
        'You will craft an article based on the planner’s outline, explaining the topic in detail. This includes '
        'math derivations, real-world applications, pros and cons, and suggestions for improving model efficiency. '
        'Your writing should be engaging, well-structured, and easy to understand, guiding readers through the topic '
        'with headings and subheadings.'
    ),
    allow_delegation=False,
    verbose=True
)

# Editor Agent
editor = Agent(
    role='Editor',
    goal='Refine the article to meet publication standards and ensure readability.',
    backstory=(
        'As an editor, you will review the writer’s draft to polish the language, structure, and alignment with Medium '
        'standards. Your task is to ensure clarity, correct grammar, and flow, while also enhancing engagement through '
        'a clear, consistent tone and adding any needed explanations for technical terms.'
    ),
    allow_delegation=False,
    verbose=True
)

# Planning Task
plan = Task(
    description=(
        '1. Identify and organize content areas for {topic}.\n'
        '2. Develop an article structure including:\n'
        '   a) Topic introduction\n'
        '   b) Detailed explanation with headings\n'
        '   c) Step-by-step mathematical derivation and formula explanation\n'
        '   d) Real-life applications\n'
        '   e) Pros and cons\n'
        '   f) Model efficiency improvements and tips.\n'
        '3. Perform an audience analysis, SEO research, and include relevant keywords and sources.\n'
    ),
    expected_output=(
        'A comprehensive article plan document including an outline, headings, keywords, and an audience analysis.'
    ),
    agent=planner
)

# Writing Task
write = Task(
    description=(
        '1. Based on the plan, write a detailed and engaging article on {topic}.\n'
        '2. Include:\n'
        '   a) An engaging introduction\n'
        '   b) In-depth explanation with headings\n'
        '   c) Step-by-step mathematical derivation\n'
        '   d) Real-world applications\n'
        '   e) Pros and cons\n'
        '   f) Suggestions for improving efficiency in models.\n'
        '3. Ensure SEO keywords are used naturally.\n'
        '4. Proofread for errors and ensure alignment with the brand’s voice.\n'
    ),
    expected_output='A polished Medium article in markdown format, ready for publication.',
    agent=writer
)

# Editing Task
edit = Task(
    description=(
        '1. Edit the article for clarity, grammar, and tone.\n'
        '2. Ensure the article is well-organized and meets Medium standards.\n'
        '3. Add any necessary technical clarifications for the math derivations and applications.\n'
        '4. Confirm that the article flows naturally and is ready for publication.\n'
    ),
    expected_output='A finalized and polished article in markdown format, ready for publication.',
    agent=editor
)

# Crew for coordinating tasks
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

# Streamlit app for article generation
st.title('Article Writer')

# Input field for the blog post topic
topic = st.text_input('Enter the topic for blog post:')

# Generate button to initiate the blog generation process
if st.button('Generate the blog post'):
    if topic:
        # Start the article generation process
        result = crew.kickoff(inputs={'topic': topic})
        
        # Display the result structure
        st.write('Result structure', result)
        
        # Check if 'write' content exists in the result
        if 'write' in result:
            st.subheader(f"Blog Post on: {topic}")
            # Render the blog content in markdown format
            st.markdown(result['write'], unsafe_allow_html=True)
        else:
            st.error("No content found in result.")
    else:
        st.error("Please enter a topic.")
