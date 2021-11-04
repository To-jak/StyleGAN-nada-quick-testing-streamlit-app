import streamlit as st

import os
import sys

sys.path.append(os.path.join('./ZSSGAN'))
sys.path.append(os.path.join('./restyle'))

import numpy as np

import torch
from PIL import Image

from tqdm import tqdm
from argparse import Namespace

from ZSSGAN.model.ZSSGAN import ZSSGAN
from ZSSGAN.utils.file_utils import save_images
from ZSSGAN.utils.training_utils import mixing_noise

from st_app.sidebar import get_sidebar_params
from st_app.utils import generate_new_samples, upload_face, whitespace
from st_app.cache import restyle_encode, APP_CONFIG


st.header('Stylegan-Nada')
st.subheader('Quick testing & training interface for Stylegan2 FFHQ model')
whitespace()

# Sidebar Params
sidebar_params = get_sidebar_params()
device = APP_CONFIG['ZSGAN']['device']

# Input Image for testing on image stylegan inversion codes
input_image = upload_face() if sidebar_params['image_inversion'] else None

# Frozen generator class
if 'source' not in st.session_state:
    st.session_state.source = 'Photo'
source_class = st.text_input('Source class', key='source')

# Trained generator class
if 'target' not in st.session_state:
    st.session_state.target = 'Sketch'
target_class = st.text_input('Target class', key='target')

# Model & Training parameters
model_choice = ["ViT-B/32", "ViT-B/16"]
model_weights = [1.0, 0.0]

improve_shape = sidebar_params['improve_shape']
if improve_shape:
    model_weights[1] = 1.0
    
mixing = 0.9 if improve_shape else 0.0

auto_layers_k = int(2 * (2 * np.log2(1024) - 2) / 3) if improve_shape else 0
auto_layer_iters = 1 if improve_shape else 0

# Optionnal target_image_dir
if sidebar_params['use_target_img_dir']:
    target_img_list = [os.path.join('target_img_dir', filename) for filename in os.listdir('target_img_dir')]
    target_img_list = None if len(target_img_list) == 0 else target_img_list
else:
    target_img_list = None

training_args = {
    "size": APP_CONFIG['ZSGAN']['size'],
    "batch": 2,
    "n_sample": 4,
    "output_dir": APP_CONFIG['output_dir'],
    "lr": 0.002,
    "frozen_gen_ckpt": APP_CONFIG['ZSGAN']['frozen_gen_ckpt'],
    "train_gen_ckpt": APP_CONFIG['ZSGAN']['train_gen_ckpt'],
    "iter": sidebar_params['training_iterations'],
    "source_class": source_class,
    "target_class": target_class,
    "lambda_direction": 1.0,
    "lambda_patch": 0.0,
    "lambda_global": 0.0,
    "lambda_texture": 0.0,
    "lambda_manifold": 0.0,
    "auto_layer_k": auto_layers_k,
    "auto_layer_iters": auto_layer_iters,
    "auto_layer_batch": 8,
    "output_interval": 50,
    "clip_models": model_choice,
    "clip_model_weights": model_weights,
    "mixing": mixing,
    "phase": None,
    "sample_truncation": 0.7,
    "save_interval": sidebar_params['save_interval'],
    "target_img_list": target_img_list,
    "img2img_batch": 16,
    "device": device
}
args = Namespace(**training_args)

# Set up output directories.
sample_dir = os.path.join(args.output_dir, "sample")
APP_CONFIG['sample_dir'] = sample_dir
ckpt_dir   = os.path.join(args.output_dir, "checkpoint")
APP_CONFIG['ckpt_dir'] = ckpt_dir

os.makedirs(sample_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

torch.manual_seed(sidebar_params['seed'])
np.random.seed(sidebar_params['seed'])


# Model Initialization
if 'last_initialized_net' not in st.session_state:
    net = ZSSGAN(args)
    st.session_state.last_initialized_net = net
    g_optim = None
    st.session_state.iter_count = 0
else:
    net = st.session_state.last_initialized_net
    net.source_class = st.session_state.source
    net.target_class = st.session_state.target


# Buttons
def on_train_button():
    st.session_state.has_to_train = True
def on_reset_model_button():
    del st.session_state.last_initialized_net
def on_save_button():
    torch.save(
        {
            "g_ema": net.generator_trainable.generator.state_dict()
        },
        f"{ckpt_dir}/{net.source_class}_{net.target_class}_{str(st.session_state.iter_count).zfill(6)}.pt",
    )
col1, col2, col3 = st.columns(3)
col1.button('Train', on_click=on_train_button)
col2.button('new model & source/target', on_click=on_reset_model_button)
col3.button('Save model', on_click=on_save_button)


# Training
if 'has_to_train' not in st.session_state:
    st.session_state.has_to_train = False

if st.session_state.has_to_train:
    print(net.source_class)
    print(net.target_class)
    g_reg_ratio = 4 / 5
    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    # Training loop
    fixed_z = torch.randn(args.n_sample, 512, device=device)
    progress_bar = st.progress(0)
    for i in tqdm(range(args.iter)):
        net.train()
            
        sample_z = mixing_noise(args.batch, 512, args.mixing, device)

        [sampled_src, sampled_dst], clip_loss = net(sample_z)

        net.zero_grad()
        clip_loss.backward()

        g_optim.step()

        if i % sidebar_params['output_interval'] == 0:
            net.eval()

            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net([fixed_z], truncation=args.sample_truncation)

                grid_rows = 4

                save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

                img = Image.open(os.path.join(sample_dir, f"dst_{str(i).zfill(6)}.jpg")).resize((1024, 256))
                st.image(img)
        
        if (args.save_interval > 0) and (i > 0) and (i % args.save_interval == 0):
            torch.save(
                {
                    "g_ema": net.generator_trainable.generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )
        st.session_state.iter_count += 1
        progress_bar.progress(int(i/args.iter*100))
    
    progress_bar.progress(100)
    # Save net in session state
    st.session_state.last_initialized_net = net
    st.session_state.has_to_train = False


# New Samples Vizualization
if sidebar_params['samples_grid']:
    st.subheader('Samples Grid')
    truncation = sidebar_params['truncation']
    samples = 9
    generate_new_samples(net, samples, truncation, APP_CONFIG)


# Encode Input Image
if input_image is not None:
    st.subheader('Image inversion for editing')
    inverted_latent = restyle_encode(input_image)
    with torch.no_grad():
        net.eval()
        
        [sampled_src, sampled_dst] = net(inverted_latent.to(device), input_is_latent=True)[0]
        
        joined_img = torch.cat([sampled_src, sampled_dst], dim=0)
        save_images(joined_img, sample_dir, "joined", 2, 0)
        st.image(Image.open(os.path.join(sample_dir, f"joined_{str(0).zfill(6)}.jpg")).resize((512, 256)))