import chainlit as cl
import requests
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from environment variables
hf_token = os.getenv("HUGGING_FACE_TOKEN")

# Ensure Hugging Face token is present
if not hf_token:
    raise Exception("Hugging Face Token not found in .env file.")

# Function to call the Hugging Face Inference API for text-to-video
def generate_video(prompt):
    api_url = "https://api-inference.huggingface.co/models/damo-vilab/modelscope-damo-text-to-video-synthesis"
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    try:
        # Send the request to the Hugging Face Inference API
        response = requests.post(api_url, headers=headers, json={"inputs": prompt})

        # Check for response status code
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            raise Exception(f"Failed to generate video: {response.text}")

        return response.content

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error in connecting to Hugging Face API: {e}")

# Chainlit message handler
@cl.on_message
async def on_message(message: cl.Message):
    try:
        # Extract the text content from the message object
        prompt_text = message.content

        # Generate the video from the prompt
        video_bytes = generate_video(prompt_text)

        # Save the video bytes to a temporary file
        video_file_path = "generated_video.mp4"
        with open(video_file_path, "wb") as f:
            f.write(video_bytes)

        # Send the video back to the Chainlit interface
        await cl.Message(content="Here is the generated video", file=video_file_path).send()

    except Exception as e:
        # Send an error message using cl.Message
        await cl.Message(content=f"Error: {e}").send()

# Testing network connection and token
if hf_token is not None:
    print("Hugging Face token loaded successfully.")
else:
    print("Failed to load Hugging Face token.")
