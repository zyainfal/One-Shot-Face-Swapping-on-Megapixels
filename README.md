![title](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/imgs/title.PNG)

![snapshot](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/imgs/snapshot1.png)

<div align="right">In CVPR 2021</div>

## Abstract 

><div align="justify">Face swapping has both positive applications such as entertainment, human-computer interaction, etc. and negative applications such as DeepFake threats to politics, economics, etc. Nevertheless, it is necessary to understand the scheme of advanced methods for high-quality face swapping and generate enough and representative face swapping images to train DeepFake detection algorithms. This paper proposes the first Megapixel level method for one shot Face Swapping (or MegaFS for short). Firstly, MegaFS organizes face representation hierarchically by the proposed Hierarchical Representation Face Encoder (HieRFE) in an extended latent space to maintain more facial details, rather than compressed representation in previous face swapping methods. Secondly, a carefully designed Face Transfer Module (FTM) is proposed to transfer the identity from a source image to the target by a non-linear trajectory without explicit feature disentanglement. Finally, the swapped faces can be synthesized by StyleGAN2 with the benefits of its training stability and powerful generative capability. Each part of MegaFS can be trained separately so the requirement of our model for GPU memory can be satisfied for megapixel face swapping. In summary, complete face representation, stable training, and limited memory usage are the three novel contributions to the success of our method. Extensive experiments demonstrate the superiority of MegaFS and the first megapixel level face swapping database is released for research on DeepFake detection and face image editing in the public domain.</div>

------
## How to Use
Please refer inference [README.md](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/inference/README.md) for more details.

## MegaFS Dataset

This dataset is built for ***forgery detection*** and ***face swapping*** research, as test dataset and comparative results respectively.

Based on [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans), we have swapped faces on ![](http://latex.codecogs.com/svg.latex?1024\times1024) and divided them into three subsets according to the three swapping modules designed in our paper. 

Following the license of [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans), the dataset is made freely available for non-commercial use under the Creative Commons [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license. 

| Swapping Module    | Number of Images |
| ------------------ | ---------------- |
| MegaFS-LCR         | 30010            |
| MegaFS-IDInjection | 30441            |
| MegaFS-FTM         | 30038            |

Each image is named as **targetID_sourceID.jpg**, where ID represents the image offset in CelebA-HQ, source image provides the identity and target image provides other information.

## Dataset Download Links
The data includes swapped faces on CelebA-HQ by MegaFS(FTM, ID injection, LCR).

In addition, `MegaFS-FF++.zip` contains the swapped faces in frames (by MegaFS 256px version) and the 889 ID folders for future research.

* [~~Google Drive~~](https://drive.google.com/drive/folders/1K6114RZv6goY-8xuxQmSamcrW2i29nG7?usp=sharing)
* [Baidu Cloud](https://pan.baidu.com/s/19vRj6jPtzxkDm2h7vFXf4w) (access code: 6bbh)

## Citation
You can find the paper at this [link](http://arxiv.org/abs/2105.04932).

If you found this dataset is helpful, please cite:
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
