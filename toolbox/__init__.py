import logging
import torch
import matplotlib.pyplot as plt

from toolbox.parameters import Experiment_parameters
from toolbox.optimizers import get_optimizer_scheduler
from toolbox.experiment import Experiment
from toolbox.image_preprocessing import tensor_to_image
from toolbox.plotter import save_plot

def get_experiment_parameters(args):
    return Experiment_parameters(args)

def get_optimizer(experience,parameters,losses = None):
    return get_optimizer_scheduler(experience,parameters,losses = losses)

def configure_logger(path):
    logging.basicConfig(filename=path,format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)

def get_experiment(parameters):
    return Experiment(parameters)

def save_all(experiment,model,parameters,listener):

    # saving experiment
    experiment.save(parameters.save_experiment_path)

    # saving exp parameters
    parameters.save()

    # saving the model
    torch.save(model, parameters.save_model_path)
    # WE SHOULD STILL UPDATE THE VALUES OF THE LOSSES WITH THEIR SAVED ONES
        
    # saving the listener
    listener.save(parameters.save_listener_path)

def generate_plots(parameters, listener):
    save_plot(parameters, listener,tags=['train'], name='style_score', title='Evolution of the style loss over epochs')
    save_plot(parameters, listener,tags=['train'], name='content_score', title='Evolution of the content loss over epochs')
    save_plot(parameters, listener,tags=['train'], name='reg_score', title='Evolution of the regularization loss over epochs')
    save_plot(parameters, listener,tags=['train'], name='total_score', title='Evolution of the total loss over epochs')
    save_plot(parameters, listener,tags=['train'], name='epoch_time', title='Evolution of the epoch time over epochs')
    save_plot(parameters, listener,tags=['train'], name='lr', title='Evolution of the learning rate over epochs')


def save_images(
    path,
    style_img,
    output_img,
    content_img,
    style_title="Style Image",
    output_title="Output Image",
    content_title="Content Image",
):
    """
    Plots style, output and content images to ease comparison.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_image(style_img))
    plt.title("Style Image")

    plt.subplot(1, 3, 2)
    plt.imshow(tensor_to_image(output_img))
    plt.title("Output Image")

    plt.subplot(1, 3, 3)
    plt.imshow(tensor_to_image(content_img))
    plt.title("Content Image")

    plt.tight_layout()
    plt.savefig(path)