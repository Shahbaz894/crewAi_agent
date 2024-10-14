import chainlit as cl
from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Define agents
planner = Agent(
    role='Content Planner',
    goal='Plan engaging and factually accurate content on {topic}.',
    backstory='You are working on planning a blog article about the topic: {topic}. '
              'You collect information that helps the audience learn something and make informed decisions. '
              'Your work is the basis for the content writer to write an article on this topic.',
    allow_delegation=False,
    verbose=True  # Ensure verbose is a boolean
)

writer = Agent(
    role='Content Writer',
    goal='Write an insightful and factually accurate opinion piece about the topic: {topic}.',
    backstory="You're working on writing a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of the Content Planner, who provides an outline and relevant context "
              "about the topic. You also provide objective and impartial insights, backing them up with information "
              "provided by the Content Planner.",
    allow_delegation=False,
    verbose=True  # Ensure verbose is a boolean
)

editor = Agent(
    role='Editor',
    goal='Edit the writing style to match the organization’s tone.',
    backstory="You are an editor who receives a blog post from the Content Writer. "
              "Your goal is to review the blog post to ensure it follows journalistic best practices, "
              "provides balanced viewpoints when providing opinions, and avoids major controversial topics or opinions.",
    allow_delegation=False,
    verbose=True  # Ensure verbose is a boolean
)

# Define tasks
plan = Task(
    description=(
        '1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n'
        '2. Identify the target audience, considering their interests and pain points.\n'
        '3. Develop a detailed content outline including an introduction, key points, and a call to action.\n'
        '4. Include SEO keywords and relevant data or sources.'
    ),
    expected_output='A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.',
    agent=planner,
)

write = Task(
    description=(
        '1. Use the content plan to craft a compelling blog post on {topic}.\n'
        '2. Incorporate SEO keywords naturally.\n'
        '3. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.\n'
        '4. Proofread for grammatical errors and alignment with the brand\'s voice.'
    ),
    expected_output='A well-written blog post in markdown format, ready for publication.',
    agent=writer,
)

edit = Task(
    description="Proofread the given blog post for grammatical errors and alignment with the brand's voice.",
    expected_output="A polished blog post in markdown format, ready for publication.",
    agent=editor,
)

# Define crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True  # Changed to a boolean
)

# Chainlit chatbot UI
@cl.on_message
def main(message: str):
    topic = message.strip()
    # Run the crew AI with user input
    result = crew.kickoff(inputs={"topic": topic})
    # Output the result to Chainlit UI
    cl.Message(content=f"Generated content on topic '{topic}':\n{result}").send()
