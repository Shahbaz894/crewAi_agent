import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from environment variables
hf_token = os.getenv("HUGGING_FACE_TOKEN")


# Function to call the Hugging Face Inference API
def generate_image_from_api(prompt):
    api_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }
    
    # Send the request to the Hugging Face Inference API
    response = requests.post(api_url, headers=headers, json={"inputs": prompt})
    
    if response.status_code != 200:
        raise Exception(f"Failed to generate image: {response.text}")
    
    return response.content

# Streamlit UI
st.title("Text-to-Image Generation with Hugging Face Inference API")

prompt = st.text_input("Enter your text prompt:")

if prompt:
    with st.spinner('Generating image...'):
        try:
            # Call the Hugging Face API to generate the image
            image_bytes = generate_image_from_api(prompt)
            
            # Display the generated image in Streamlit
            image = Image.open(BytesIO(image_bytes))
            st.image(image, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

st.write("Powered by Hugging Face Inference API")
