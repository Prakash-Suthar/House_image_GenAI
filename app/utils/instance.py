import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch import autocast

from transformers import CLIPTextModel, CLIPTokenizer,CLIPTokenizerFast
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from transformers import DPTImageProcessor

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from depth2img import Depth2ImgPipeline

from transformers import AutoTokenizer


device = 'cpu'
# Load autoencoder
vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='vae').to(device)
# Load tokenizer and the text encoder
# tokenizer = AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='tokenizer')

tokenizer = CLIPTokenizerFast.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='tokenizer')
print("tokenssssssss",tokenizer)
text_encoder = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='text_encoder').to(device)
# Load UNet model
unet = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='unet').to(device)
# Load scheduler
scheduler = PNDMScheduler(beta_start=0.00085,
                        beta_end=0.012,
                        beta_schedule='scaled_linear',
                        num_train_timesteps=1000)
# Load DPT Depth Estimator
depth_estimator = DPTForDepthEstimation.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='depth_estimator')
# Load DPT Feature Extractor
# depth_feature_extractor = DPTFeatureExtractor.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='feature_extractor')
depth_feature_extractor = DPTImageProcessor.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='feature_extractor')
depth2img = Depth2ImgPipeline(vae,
                            tokenizer,
                            text_encoder,
                            unet,
                            scheduler,
                            depth_feature_extractor,
                            depth_estimator)



# img = "./assests/input_img/master1 (1).png"
img = r"E:\codified\gen_ai\House_image_genAI\app\assests\input_img\master1 (1).png"
im = Image.open(img)
# im.show()

prompt = "colourfull plantation on windows"

depth2img("colourful building", im)[0]


