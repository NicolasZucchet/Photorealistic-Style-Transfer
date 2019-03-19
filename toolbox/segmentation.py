import matplotlib.image as mpimg
import torch
import pandas

from toolbox.image_preprocessing import image_loader
from toolbox.path_setup import *
import matplotlib.pyplot as plt

def generate_segmentation_cli(name_experiment, images):
    experiments_path, images_path, results_path, masks_path = prepare_experiment(name_experiment)

    to_main = "../"
    string_paths = string_images(images, images_path, prefix=to_main)

    # generation of the cli
    cli = "test.py \
          --model_path models/baseline-resnet50dilated-ppm_deepsup \
          --test_imgs " + string_paths + " \
          --arch_encoder resnet50dilated \
          --arch_decoder ppm_deepsup \
          --fc_dim 2048 \
          --result " + to_main + results_path

    return cli, masks_path, results_path

def save_segmentation(image_path, save_dir):
    print(image_path, save_dir)
    img = mpimg.imread(image_path)
    width = img.shape[1]
    new = img[:, int(width / 2):]
    save_path = save_dir + "/" + image_path.split("/")[-1]
    mpimg.imsave(save_path, new)

def plot_segmented_images(path, images):
    plt.figure(figsize=(20,10))
    columns = 2
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        img=mpimg.imread(path + "/" + image[:-4]+"_seg.png")
        # separate the image into input value and segmented value
        plt.imshow(img)



colormat = scipy.io.loadmat('semsegpt/data/color150.mat')
RGBcolors = np.array(colormat['colors'])
HEXcolors = [rgb_to_hexa(i) for i in RGBcolors]
df = pandas.read_csv('semsegpt/data/object150_info.csv')
objects = list(df.Name)

class Counter:
    def __init__(self):
        self.dict = {}

    def add(self, e):
        if e in self.dict:
            self.dict[e] += 1
        else:
            self.dict[e] = 1

    def top(self, k):
        i = 0
        r = []
        s = [k for k in sorted(self.dict, key=self.dict.get, reverse=True)]
        for key in s:
            r += [key]
            i += 1
            if i > k:
                break
        return r

def get_id_from_HEXcolor(HEXcolor):
    for i in range(len(HEXcolors)):
        if (HEXcolors[i] == HEXcolor).all():
            return i
    return -1

def get_id_from_RGBcolor(RGBcolor):
    for i in range(len(RGBcolors)):
        if (RGBcolors[i] == RGBcolor).all():
            return i
    return -1

def get_all_topics(segmented_image, k=3):
    m = len(segmented_image)
    n = len(segmented_image[0])
    c = Counter()
    for i in range(m):
        for j in range(n):
            hex = rgb_to_hexa(segmented_image[i][j] * 255)
            c.add(hex)
    topics = ""
    for e in c.top(k):
        id = get_id_from_HEXcolor(e)
        topics += objects[id] + ";"
    return topics

water_class = [21, 26, 60, 128]
tree_class = [4, 9, 17, 32]
building_class = [0, 1, 25, 48, 79, 84]
road_class = [6, 11, 13, 29, 46, 51, 52, 68, 91, 94, 101]
roof_class = [5, 86]
mountain_class = [16, 34]
stair_class = [53, 59, 121]
chair_class = [19, 23, 30, 31, 69, 75]
vehicle_class = [20, 80, 83, 102]

merge_classes = [water_class, tree_class, building_class,
                 road_class, roof_class, mountain_class,
                 stair_class, chair_class, vehicle_class]

del_classed = [26, 60, 128, 9, 17, 32, 1, 25, 48, 79, 84, 11, 13, 29, 46, 51, 52, 68, 91, 94, 101,
               86, 34, 59, 121, 23, 30, 31, 69, 75, 80, 83, 102]


def get_segmentation(segmentation_path,size):
    seg_result = image_loader(segmentation_path,size).squeeze(0)
    c,w,h = seg_result.size()
    masks = torch.zeros(150,w,h, dtype=torch.int)
    for i in range(w):
        for j in range(h):
            id = get_id_from_RGBcolor(seg_result[:3,i,j].data.numpy())
            masks[id,i,j] = 1
    channels, height_, width_ = masks.size()


    for classes in merge_classes:
        for index, each_class in enumerate(classes):
            if index == 0:
                zeros_index = each_class
                base_map = masks[each_class, :, :].clone()
            else:
                base_map = base_map | masks[each_class, :, :]
        masks[zeros_index, :, :] = base_map

    return masks, height_, width_

def merge_mask(style_mask_origin,content_mask_origin,height_,width_,height2,width2,device):
    merged_style_mask = np.zeros((117, height_, width_), dtype='int')
    merged_content_mask = np.zeros((117, height2, width2), dtype='int')
    print()
    # --------------------------
    count = 0
    for i in range(150):
        temp = style_mask_origin[i, :, :].numpy()
        if i not in del_classed and np.sum(temp) > 50:
            # print(count, np.sum(temp))
            merged_style_mask[count, :, :] = temp
            merged_content_mask[count, :, :] = content_mask_origin[i, :, :].numpy()
            count += 1
        else:
            pass
    style_mask_tensor = torch.from_numpy(merged_style_mask[:count, :, :]).float().to(device)
    content_mask_tensor = torch.from_numpy(merged_content_mask[:count, :, :]).float().to(device)
    return style_mask_tensor,content_mask_tensor