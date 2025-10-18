
import os
from dotenv import load_dotenv
import logging
from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool
from phi.llm.openai import OpenAIChat
from crewai import Agent, Task, Crew
import chainlit as cl  # Import Chainlit for app integration
import phi.utils
print(dir(phi.utils))

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

suport_agent=Agent(
    
    role="Sewior Support Representative",
    goal="Be the most friendly and helpful",
    backstory=(
        "You work at crewAI (https://crewai.com) and "
        " are now working on providing "
		"support to {customer}, a super important customer "
        " for your company."
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions "
        
    ),
    allow_delegation=False,
    verbose=False
    
    
)

suport_quality_assurance_agent=Agent(
    role="Support Quality Aussurance Specialist",
    goal="Get recogination for providing the"
    "best support quality assurance in your team",
    backstory=("You work at crewAI (https://crewai.com) and "
        " are now working on providing "
		"support to {customer}, a super important customer "
        " for your company."
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions"
)
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
)


inquiry_resolation=Task(
    description=("{customer} just reached out with super important ask:\n"
                 "{inquiry}\n\n"
                 "{person} from {customer} is the one that reached out."
                 "Make sure to use everything you know "
                "to provide the best support possible."
                "You must strive to provide a complete "
                "and accurate response to the customer's inquiry."
                 ),
    expected_output=(
        "A detailed, informative response to the "
        "customer's inquiry that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
    tools=[docs_scrape_tool],
    agent=suport_agent
)

suport_quality_assurance_review=Task(
    description=("Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
		"high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
		"thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        " find the information, "
		"ensuring the response is well-supported and "
        "leaves no questions unanswered."),
    expected_output=( "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
		"relevant feedback and improvements.\n"
		"Don't be too formal, we are a chill and cool company "
	    "but maintain a professional and friendly tone throughout."),
    
   agent=suport_quality_assurance_agent 
)
crew=Crew(
    agents=[suport_agent,suport_quality_assurance_agent],
    tasks=[inquiry_resolation,suport_quality_assurance_review],
    verbose=1,
    memory=True
)

inputs={
    "customer": "CampusX",
    "person": "Md Amanatullah",
    "inquiry": "I need help with setting up a Crew "
               "and kicking it off, specifically "
               "how can I add memory to my crew? "
               "Can you provide guidance?"
}
result = crew.kickoff(inputs=inputs)
print(result)