import matplotlib.image as mpimg
import wget
import pandas
from toolbox.path_setup import *
import matplotlib.pyplot as plt

def generate_segmentation(name_experiment, images):
    experiments_path, images_path, results_path, masks_path = prepare_experiment(name_experiment)

    to_experiments = "../" + experiments_path
    string_paths = string_images(images, images_path, prefix=to_experiments)

    # generation of the cli
    cli = "test.py \
          --model_path models \
          --test_imgs " + string_paths + " \
          --arch_encoder resnet50dilated \
          --arch_decoder ppm_deepsup \
          --fc_dim 2048 \
          --result " + to_experiments + results_path

    os.chdir("semsegpt")
    os.system("python "+cli)
    os.chdir("..")

    for image in images:
        res_path = results_path + "/" + image[:-4] + '.png'
        save_segmentation(res_path, masks_path)

    return "experiments/" + masks_path, results_path

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
        img=mpimg.imread(path + "/" + image[:-4]+".png")
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

def get_id_from_color(HEXcolor):
    for i in range(len(HEXcolors)):
        if HEXcolors[i] == HEXcolor:
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
        id = get_id_from_color(e)
        print(id)
        topics += objects[id] + ";"
    return topics
