import numpy as np
import yaml
import argparse
import math
import torch
import torch.nn as nn
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from denoising_diffusion_pytorch.uncond_unet import Unet
from denoising_diffusion_pytorch.data import *
from fvcore.common.config import CfgNode
from einops import rearrange
from PIL import Image

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf
def parse_args():
    parser = argparse.ArgumentParser(description="demo configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, default="./configs/default.yaml")
    parser.add_argument("--pre_weight", help='path of pretrained weight', type=str, required=True)
    parser.add_argument("--sampling_timesteps", help='sampling timesteps', type=int, default=1)
    parser.add_argument("--bs", help='batch_size for inference', type=int, default=8)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args

def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )
    if model_cfg.model_name == 'cond_unet':
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.get('learned_variance', False),
                    out_mul=unet_cfg.out_mul,
                    cond_in_dim=unet_cfg.cond_in_dim,
                    cond_dim=unet_cfg.cond_dim,
                    cond_dim_mults=unet_cfg.cond_dim_mults,
                    window_sizes1=unet_cfg.window_sizes1,
                    window_sizes2=unet_cfg.window_sizes2,
                    fourier_scale=unet_cfg.fourier_scale,
                    cfg=unet_cfg,
                    )
    else:
        raise NotImplementedError
    if model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
    
    model = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    cfg.sampler.ckpt_path = args.pre_weight
    cfg.sampler.batch_size = args.bs

    data = torch.load(cfg.sampler.ckpt_path, map_location="cpu")
    if cfg.sampler.use_ema:
        sd = data['ema']
        new_sd = {}
        for k in sd.keys():
            if k.startswith("ema_model."):
                new_k = k[10:]  # remove ema_model.
                new_sd[new_k] = sd[k]
        sd = new_sd
        model.load_state_dict(sd)
    else:
        model.load_state_dict(data['model'])
    if 'scale_factor' in data['model']:
        model.scale_factor = data['model']['scale_factor']

    model.eval().cuda()
    image = Image.open("/content/input.png").convert("RGB")
    image = rearrange(torch.from_numpy(np.array(image)), "h w c -> 1 c h w").float() / 255.
    image = normalize_to_neg_one_to_one(image)
    image = image.cuda()
    return sample(model, image, cfg, batch_size=cfg.sampler.batch_size)

def sample(model, image, cfg, batch_size=8):
    mask = None
    if cfg.sampler.sample_type == 'whole':
        batch_pred = whole_sample(model, image, raw_size=image.shape[2:], mask=mask)
    elif cfg.sampler.sample_type == 'slide':
        batch_pred = slide_sample(model, image, crop_size=cfg.sampler.get('crop_size', [320, 320]),
                                        stride=cfg.sampler.stride, mask=mask, bs=batch_size)
    print(batch_pred.shape)
    tv.utils.save_image(batch_pred, "/content/result.png")

def whole_sample(model, inputs, raw_size, mask=None):
    inputs = F.interpolate(inputs, size=(416, 416), mode='bilinear', align_corners=True)
    seg_logits = model.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
    seg_logits = F.interpolate(seg_logits, size=raw_size, mode='bilinear', align_corners=True)
    return seg_logits

def slide_sample(model, inputs, crop_size, stride, mask=None, bs=8):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        # aux_out1 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        # aux_out2 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        crop_imgs = []
        x1s = []
        x2s = []
        y1s = []
        y2s = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                crop_imgs.append(crop_img)
                x1s.append(x1)
                x2s.append(x2)
                y1s.append(y1)
                y2s.append(y2)
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_seg_logits_list = []
        num_windows = crop_imgs.shape[0]
        bs = bs
        length = math.ceil(num_windows / bs)
        for i in range(length):
            if i == length - 1:
                crop_imgs_temp = crop_imgs[bs * i:num_windows, ...]
            else:
                crop_imgs_temp = crop_imgs[bs * i:bs * (i + 1), ...]

            crop_seg_logits = model.sample(batch_size=crop_imgs_temp.shape[0], cond=crop_imgs_temp, mask=mask)
            crop_seg_logits_list.append(crop_seg_logits)
        crop_seg_logits = torch.cat(crop_seg_logits_list, dim=0)
        for crop_seg_logit, x1, x2, y1, y2 in zip(crop_seg_logits, x1s, x2s, y1s, y2s):
            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        return seg_logits

if __name__ == "__main__":
    args = parse_args()
    main(args)