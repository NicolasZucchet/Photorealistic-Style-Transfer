import os
import wget
import scipy.io
import numpy as np

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def download_models():
    # path for models
    models_path = 'semsegpt/models'
    model = 'baseline-resnet50dilated-ppm_deepsup'
    model_path = models_path + "/" + model
    encoder = model + '/encoder_epoch_20.pth'
    encoder_path = models_path + "/" + encoder
    decoder = model + '/decoder_epoch_20.pth'
    decoder_path = models_path + "/" + decoder

    print(models_path)

    # creating dirs
    to_create = [models_path, model_path]
    for folder in to_create:
        create_path(folder)
    if not os.path.exists(encoder_path):
        print("plse")
        url = "http://sceneparsing.csail.mit.edu/model/pytorch/" + encoder
        wget.download(url, out=model_path)
    if not os.path.exists(decoder_path):
        url = "http://sceneparsing.csail.mit.edu/model/pytorch/" + decoder
        wget.download(url, out=model_path)

def prepare_experiment(name_experiment):
    experiments_path = "examples"
    # path for experiment
    experiment_path = experiments_path + "/" + name_experiment
    # path from images
    images_path = experiment_path + '/images'
    # path for saving results
    results_path = experiment_path + '/images'
    masks_path = experiment_path + '/images'

    # creating dirs
    to_create = [experiments_path, experiment_path, images_path, results_path, masks_path]
    for folder in to_create:
        create_path(folder)

    return experiments_path, images_path, results_path, masks_path

def download_image(url,path):
    if not os.path.exists(path):
        wget.download(url, out=path)

def get_path_images(names, path):
    path_images = []
    for name in names:
        path_images += [path + "/" + name]
    return path_images

def string_images(img_names, images_path, prefix=""):
    path_images = get_path_images(img_names, images_path)
    string_paths = ""
    for i in range(len(path_images) - 1):
        string_paths += (prefix + path_images[i] + " ")
    string_paths += prefix + path_images[-1]
    return string_paths

def rgb_to_hexa(color):
  # transforms to hexadecimal
  return '%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))

mat = scipy.io.loadmat('semsegpt/data/color150.mat')
RGBcolors = np.array(mat['colors'])
HEXcolors = [rgb_to_hexa(i) for i in RGBcolors]