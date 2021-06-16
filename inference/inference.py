import argparse
import os
import cv2
from megafs import resnet50, HieRFE, Generator, FaceTransferModule
import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as tF

def encode_segmentation_rgb(segmentation, no_neck=True):
    parse = segmentation[:,:,0]

    face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse==mouth_id)
    mouth_map[valid_index] = 255
    valid_index = np.where(parse==hair_id)
    hair_map[valid_index] = 255

    return np.stack([face_map, mouth_map, hair_map], axis=2)

class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask

class MegaFS(object):
    def __init__(self, swap_type, img_root, mask_root):
        # Inference Parameters
        self.size = 1024
        self.swap_type = swap_type
        self.img_root = img_root
        self.mask_root = mask_root

        # Model
        # "ftm"    "injection"     "lcr"
        num_blocks = 3 if self.swap_type == "ftm" else 1
        latent_split = [4, 6, 8]
        num_latents = 18
        swap_indice = 4 
        self.encoder = HieRFE(resnet50(False), num_latents=latent_split, depth=50).cuda()
        self.swapper = FaceTransferModule(num_blocks=num_blocks, swap_indice=swap_indice, num_latents=num_latents, typ=self.swap_type).cuda()
        ckpt_e = "./checkpoint/{}_final.pth".format(self.swap_type)
        if ckpt_e is not None:
            print("load encoder & swapper:", ckpt_e)
            ckpts = torch.load(ckpt_e, map_location=torch.device("cpu"))
            self.encoder.load_state_dict(ckpts["e"])
            self.swapper.load_state_dict(ckpts["s"])
            del ckpts

        self.generator = Generator(self.size, 512, 8, channel_multiplier=2).cuda()
        ckpt_f = "./checkpoint/stylegan2-ffhq-config-f.pth"
        if ckpt_f is not None:
            print("load generator:", ckpt_f)
            ckpts = torch.load(ckpt_f, map_location=torch.device("cpu"))
            self.generator.load_state_dict(ckpts["g_ema"], strict=False)
            del ckpts

        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
        
        self.encoder.eval()
        self.swapper.eval()
        self.generator.eval()
    
    def read_pair(self, src_idx, tgt_idx):
        src_face = cv2.imread(os.path.join(self.img_root, "{}.jpg".format(src_idx)))
        tgt_face = cv2.imread(os.path.join(self.img_root, "{}.jpg".format(tgt_idx)))
        tgt_mask  = cv2.imread(os.path.join(self.mask_root, "{}.png".format(tgt_idx)))

        src_face_rgb = src_face[:, :, ::-1]
        tgt_face_rgb = tgt_face[:, :, ::-1]
        tgt_mask = encode_segmentation_rgb(tgt_mask)
        return src_face_rgb, tgt_face_rgb, tgt_mask

    def preprocess(self, src, tgt):
        src = cv2.resize(src.copy(), (256, 256))
        tgt = cv2.resize(tgt.copy(), (256, 256))
        src = torch.from_numpy(src.transpose((2, 0, 1))).float().mul_(1/255.0)
        tgt = torch.from_numpy(tgt.transpose((2, 0, 1))).float().mul_(1/255.0)

        src = tF.normalize(src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
        tgt = tF.normalize(tgt, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)

        return src.unsqueeze_(0), tgt.unsqueeze_(0)

    def run(self, src_idx, tgt_idx, refine=True):
        src_face_rgb, tgt_face_rgb, tgt_mask = self.read_pair(src_idx, tgt_idx)
        source, target = self.preprocess(src_face_rgb, tgt_face_rgb)
        swapped_face = self.swap(source, target)
        swapped_face = self.postprocess(swapped_face, tgt_face_rgb, tgt_mask)

        result = np.hstack((src_face_rgb[:,:,::-1], tgt_face_rgb[:,:,::-1], swapped_face))

        if refine:
            swapped_tensor, _ = self.preprocess(swapped_face[:,:,::-1], swapped_face)
            refined_face = self.refine(swapped_tensor)
            refined_face = self.postprocess(refined_face, tgt_face_rgb, tgt_mask)
            result = np.hstack((result, swapped_face))
        cv2.imwrite("{}.jpg".format(self.swap_type), result)

    def swap(self, source, target):
        with torch.no_grad():
            ts = torch.cat([target, source], dim=0).cuda()
            lats, struct = self.encoder(ts)

            idd_lats = lats[1:]
            att_lats = lats[0].unsqueeze_(0)
            att_struct = struct[0].unsqueeze_(0)

            swapped_lats = self.swapper(idd_lats, att_lats)
            fake_swap, _ = self.generator(att_struct, [swapped_lats, None], randomize_noise=False)

            fake_swap_max = torch.max(fake_swap)
            fake_swap_min = torch.min(fake_swap)
            denormed_fake_swap = (fake_swap[0] - fake_swap_min) / (fake_swap_max - fake_swap_min) * 255.0
            fake_swap_numpy = denormed_fake_swap.permute((1, 2, 0)).cpu().numpy()
        return fake_swap_numpy
    
    def refine(self, swapped_tensor):
        with torch.no_grad():
            lats, struct = self.encoder(swapped_tensor.cuda())

            fake_refine, _ = self.generator(struct, [lats, None], randomize_noise=False)

            fake_refine_max = torch.max(fake_refine)
            fake_refine_min = torch.min(fake_refine)
            denormed_fake_refine = (fake_refine[0] - fake_refine_min) / (fake_refine_max - fake_refine_min) * 255.0
            fake_refine_numpy = denormed_fake_refine.permute((1, 2, 0)).cpu().numpy()
        return fake_refine_numpy
    
    def postprocess(self, swapped_face, target, target_mask):
        target_mask = cv2.resize(target_mask, (self.size,  self.size))

        mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
        face_mask_tensor = mask_tensor[0] + mask_tensor[1]
        
        soft_face_mask_tensor, _ = self.smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
        soft_face_mask_tensor.squeeze_()

        soft_face_mask = soft_face_mask_tensor.cpu().numpy()
        soft_face_mask = soft_face_mask[:, :, np.newaxis]
        result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
        result = result[:,:,::-1].astype(np.uint8)
        return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap_type", type=str, default="ftm")
    parser.add_argument("--img_root", type=str, default="/data/yuhao.zhu/CelebA-HQ")
    parser.add_argument("--mask_root", type=str, default="/data/yuhao.zhu/CelebAMaskHQ-mask")
    parser.add_argument("--srcID", type=int, default=2332)
    parser.add_argument("--tgtID", type=int, default=2107)

    args = parser.parse_args()
    handler = MegaFS(args.swap_type, args.img_root, args.mask_root)
    handler.run(args.srcID, args.tgtID)
  
