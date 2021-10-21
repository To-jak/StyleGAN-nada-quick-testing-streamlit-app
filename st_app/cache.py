import streamlit as st

import yaml
from argparse import Namespace

import torch
from torchvision import transforms

from restyle.models.e4e import e4e
from restyle.utils.inference_utils import run_on_batch

from st_app.image_inversion.inversion_processor import InversionPreprocessor


@st.cache(allow_output_mutation=True)
def get_app_config():
    with open('st_app/app_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config
APP_CONFIG = get_app_config()


@st.cache(hash_funcs={torch.jit._script.RecursiveScriptModule: lambda x: x.original_name})
def load_inversion_processor():
    return InversionPreprocessor(APP_CONFIG['ZSGAN']['device'])
inv_processor = load_inversion_processor()


@st.cache(hash_funcs={torch.jit._script.RecursiveScriptModule: lambda x: x.original_name})
def preprocess_image(uploaded_image):
    aligned_faces = inv_processor(uploaded_image)[0]
    if len(aligned_faces) == 0:
        st.warning('No face detected')
        return None
    return aligned_faces[0]


@st.cache
def load_restyle_encoder(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    opts.device = APP_CONFIG['restyle']['device']
    restyle_net = e4e(opts)
    restyle_net.eval()

    return restyle_net, opts
restyle_net, opts = load_restyle_encoder(APP_CONFIG['restyle']['e4e_ffhq_encode_path'])
restyle_net.to(APP_CONFIG['restyle']['device'])


restyle_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to(APP_CONFIG['restyle']['device']).float().detach()
    return avg_image

@st.cache
def restyle_encode(input_image):

    restyle_net.eval()

    transformed_image = restyle_transform(input_image)
    opts.n_iters_per_batch = 5
    opts.resize_outputs = False

    with torch.no_grad():
        avg_image = get_avg_image(restyle_net)
        _, result_latents = run_on_batch(transformed_image.unsqueeze(0).to(APP_CONFIG['restyle']['device']), restyle_net, opts, avg_image)
    
    inverted_latent = torch.Tensor(result_latents[0][4]).unsqueeze(0).unsqueeze(1).cpu()
    return inverted_latent





