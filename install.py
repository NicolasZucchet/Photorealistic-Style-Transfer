import sys
import os

os.system(sys.executable + " -m pip install -r requirements.txt")
# image segmentation
if not os.path.exists("semsegpt"):
    os.system("git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git semsegpt")

# download models for segmentation
from toolbox.path_setup import download_models
print("pls")
download_models() # yyilds an error on my windows machine
print("pk")


