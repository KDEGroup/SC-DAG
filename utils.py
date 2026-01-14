import torch
import numpy as np
import random
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from fr_model import IRSE_50, MobileFaceNet, IR_152, InceptionResnetV1
import torchvision.models as models


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_fr_model(name):
    if name == 'ResNet152':
        model = models.resnet152(pretrained=True)
    else:
        raise ValueError(f'Invalid model name: {name}')
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model.cuda()


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler

th_dict = {'ResNet152':(0.1, 0.2, 0.3)}

def asr_calculation(cos_sim_scores_dict):
    # Iterate each image pair's simi-score from "simi_scores_dict" and compute the attacking success rate
    for key, values in cos_sim_scores_dict.items():
        th01, th001, th0001 = th_dict[key]
        total = len(values)
        success01 = 0
        success001 = 0
        success0001 = 0
        for v in values:
            if v > th01:
                success01 += 1
            if v > th001:
                success001 += 1
            if v > th0001:
                success0001 += 1
        print(key, " attack success(far@0.1) rate: ", success01 / total)
        print(key, " attack success(far@0.01) rate: ", success001 / total)
        print(key, " attack success(far@0.001) rate: ", success0001 / total)