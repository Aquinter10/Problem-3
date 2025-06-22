import streamlit as st
import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from utils import load_model, generate_images

st.set_page_config(page_title="Handwritten Digit Generator", layout="centered")

st.title("✍️ Generate Handwritten Digits")
st.write("Select a number between 0 and 9 and generate 5 handwritten-style images using your own trained model.")

digit = st.number_input("Select a digit:", min_value=0, max_value=9, step=1)

if 'model' not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model = load_model()

if st.button("Generate Images"):
    with st.spinner("Generating..."):
        model = st.session_state.model
        images = generate_images(model, digit, num_images=5)

        grid = make_grid(images, nrow=5, normalize=True, pad_value=1)
        img = to_pil_image(grid)

        st.image(img, caption=f"Generated images for digit '{digit}'", use_column_width=True)
