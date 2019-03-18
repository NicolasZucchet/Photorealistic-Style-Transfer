import sys
import os
import wget
from toolbox.path_setup import download_models

os.system(sys.executable + " -m pip install -r requirements.txt")
# image segmentation
os.system("git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git semsegpt")

# download models for segmentation
download_models()


