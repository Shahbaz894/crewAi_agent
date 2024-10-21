import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = 'rain1011/pyramid-flow-sd3'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("Pyramid Flow SD3 Video Generator")

st.write('Generate video based on text or image')

# Select generation type
generation_type = st.radio("Select generation type", ("Text-to-Video", "Image-to-Video"))

if generation_type == "Text-to-Video":
    user_prompt = st.text_area("Enter the prompt for video generation")
    
    if st.button('Generate Video'):
        inputs = tokenizer(user_prompt, return_tensors='pt')
        outputs = model.generate(inputs['input_ids'], max_length=512)
        st.write('Video Generation in process')
        st.video("output_video.mp4")

elif generation_type == "Image-to-Video":
    uploaded_image = st.file_uploader('Upload an image', type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None and st.button("Generate Video"):
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("Video Generation in progress ...")
        st.video("output_video_from_image.mp4")
