# Photorealistic-Style-Transfer

Amaury Sudrie, Victor Ruelle, Nicolas Zucchet for course project (MAP583, École polytechnique, France)

## Objective
Based on the https://github.com/ray075hl/DeepPhotoStyle_pytorch implementation of the Deep Photo Style Transfer paper (Luan et. al), we aim to explore new applications and modifications of deep photo styletransfer. To perform photo style transfer, semantic segmentation is used. The quality of the transfer then depends on the style photo. We are mainly concerned with how to find a good style photo with respect to user style will and we aim to automate the whole process.

## Pipeline

The pipeline we use is `new-pipeline.ipynb`. Be sure to run `python install.py` before, in order to both install dependencies and download the segmentation model.

## Credits

Papers we took inspiration from: 
 - Deep Photo Style Transfer https://arxiv.org/abs/1703.07511
 - Neural Style Transfer https://arxiv.org/abs/1508.06576
 - Automated Deep Photo Style Transger https://arxiv.org/pdf/1901.03915.pdf

Projects used:
 - https://github.com/yagudin/PyTorch-deep-photo-styletransfer
 - https://github.com/CSAILVision/semantic-segmentation-pytorch.git which is a really good project for semantic segmentation
