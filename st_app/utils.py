import streamlit as st

import os

import torch
from PIL import Image

from ZSSGAN.utils.file_utils import save_images
from st_app.cache import preprocess_image


def whitespace():
    st.markdown('#')


def upload_face():
    col1, col2 = st.columns(2)
    img_bytes = col1.file_uploader('Upload a photo of a person face for image inversion')

    if img_bytes is None:
        return None

    # Align face
    uploaded_img = Image.open(img_bytes).convert('RGB')
    aligned_img = preprocess_image(uploaded_img)
    if aligned_img is None:
        st.warning('No face has been detected in your image.')
        st.stop()
    
    col2.image(aligned_img.resize((120, 120)))
    
    return aligned_img


def generate_new_samples(net, samples, truncation, config):
    with torch.no_grad():
        net.eval()
        sample_z = torch.randn(samples, 512, device=config['ZSGAN']['device'])

        [sampled_src, sampled_dst], loss = net([sample_z], truncation=truncation)
        grid_rows = int(samples ** 0.5)

        save_images(sampled_dst, config['sample_dir'], "sampled", grid_rows, 0)

        st.image(Image.open(os.path.join(config['sample_dir'], f"sampled_{str(0).zfill(6)}.jpg")).resize((768, 768)))