# MegaFS PyTorch Inference Code

Implementation of [**One Shot Face Swapping on Megapixels**](http://arxiv.org/abs/2105.04932) in PyTorch

## Requirements

- Please refer [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) to get StyleGAN2 environment (converted model is provided in this repo)
- Python 3.6
- PyTorch 1.5.1 (it is ok that  [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) uses 1.3.1)
- CUDA 10.1/10.2

## Usage

### Dataset

Please download [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) and [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ).

Also, please note the mask label is assigned as

  | label list      |             |             |
  | --------------- | ----------- | ----------- |
  | 0: 'background' | 1: 'skin'   | 2: 'l_brow' |
  | 3: 'r_brow'     | 4: 'l_eye'  | 5: 'r_eye'  |
  | 6: 'eye_g'      | 7: 'l_ear'  | 8: 'r_ear'  |
  | 9: 'ear_r'      | 10: 'nose'  | 11: 'mouth' |
  | 12: 'u_lip'     | 13: 'l_lip' | 14: 'neck'  |
  | 15: 'neck_l'    | 16: 'cloth' | 17: 'hair'  |
  | 18: 'hat'       |             |             |
  
in case of updated CelebAMask-HQ dataset.

### Checkpoints

[Baidu Cloud](https://pan.baidu.com/s/1DPNnU9zmkEdef6WT79J5Wg) (access code: 7nov)

[Google Drive](https://drive.google.com/drive/folders/1XDakvzNHDtC7G1d1Zn8MjPbmen4LKLPw?usp=sharing)

### Inference

> python inference.py \
> 
> --swap_type [ftm/injection/lcr] \
> 
> --img_root [CelebAHQ-PATH] \
> 
> --mask_root [CelebAMaskHQ-PATH] \
> 
> --srcID [INT-NUMBER] \
> 
> --tgtID [INT-NUMBER]

The result is rearrange as *source_image, target_image, swapped_face, refined_swapped_face*, where  *refined_swapped_face* is the reconstructed version of *swapped_face*. Please refer more details in the provided codes.

## Samples

- **FTM**

![ftm](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/inference/imgs/ftm.jpg)

- **ID Injection**

![injection](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/inference/imgs/injection.jpg)

- **LCR**

![lcr](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/inference/imgs/lcr.jpg)

## License

All the material, including source code, is made freely available for non-commercial use under the Creative Commons [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license. Feel free to use any of the material in your own work, as long as you give us appropriate credit by mentioning the title and author list of our paper:

```
@inproceedings{zhu2021megafs,
  title={One Shot Face Swapping on Megapixels},
  author={Zhu, Yuhao and Li, Qi and Wang, Jian and Xu, Chengzhong and Sun, Zhenan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
  pages = {4834-4844},
  month = {June},
  year={2021}
}
```
