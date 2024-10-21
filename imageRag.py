import chainlit as cl
import requests
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from environment variables
hf_token = os.getenv("HUGGING_FACE_TOKEN")

# Function to call the Hugging Face Inference API
def generate_image(prompt):
    api_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    response = requests.post(api_url, headers=headers, json={"inputs": prompt})

    if response.status_code != 200:
        raise Exception(f"Failed to generate image: {response.text}")

    return response.content

# Chainlit message handler
@cl.on_message
async def on_message(message):
    try:
        # Generate the image from the prompt
        image_bytes = generate_image(message.content)  # Use .content for the message

        # Convert image bytes to a PIL image
        image = Image.open(BytesIO(image_bytes))

        # Save the image to a temporary file
        image.save("generated_image.png")

        # Send the image back to the Chainlit interface with the correct for_id
        await cl.Image(path="generated_image.png", name=message.content).send(for_id=message.id)

    except Exception as e:
        # Send an error message with the correct for_id
        await cl.Message(content=f"Error: {e}").send(for_id=message.id)
