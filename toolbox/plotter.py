
'''
Plotter class for plotting various things
'''
import os
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


# get plot data from logger and plot to image file
def save_plot(parameters, logger, tags=['train'], name='epoch_time', title='', labels=None, ax = None):
    var_dict = copy.copy(logger.logged)
    labels = tags if labels is None else labels

    epochs = None
    if ax is None:
        fig, ax = plt.subplots(1,1)

    for tag in tags:
        if not(tag in var_dict):
            # happens if we ever only did training or testing 
            continue
        epochs = np.array([x for x in var_dict[tag][name].keys()]) 
        curr_line = np.array([x for x in var_dict[tag][name].values()])
        tick_spacing = max(int((float(epochs[-1])-float(epochs[0]))/10),5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        if tag == "val":
            ax.plot(epochs, curr_line, "go") 
        else:
            ax.plot(epochs, curr_line) 

    ax.set_xlabel('epochs')
    ax.set_title('{} - {}'.format(title, parameters.name))
    # plt.legend(labels=labels)

    out_fn = os.path.join(parameters.res_dir, '{}_{}.png'.format(parameters.name, name))
    plt.savefig(out_fn, bbox_inches='tight', dpi=200)
    # plt.gcf().clear()
    # plt.close()
    # return ax

def save_output_(exp):
    image = copy.deepcopy(exp.content_image)
    output = exp.model(image).data.clamp(0,1).detach().numpy()[0]
    output = np.array( [ [ (output[0][x][y], output[1][x][y], output[2][x][y] ) for y in range(output.shape[2]) ] for x in range(output.shape[1]) ] )
    plt._imsave(exp.parameters.res_dir+"output.png",output)

    output = exp.content_image.detach().numpy()[0]
    output = np.array( [ [ (output[0][x][y], output[1][x][y], output[2][x][y] ) for y in range(output.shape[2]) ] for x in range(output.shape[1]) ] )
    plt._imsave(exp.parameters.res_dir+"content_image.png",output)
    
    output = exp.style_image.detach().numpy()[0]
    output = np.array( [ [ (output[0][x][y], output[1][x][y], output[2][x][y] ) for y in range(output.shape[2]) ] for x in range(output.shape[1]) ] )
    plt._imsave(exp.parameters.res_dir+"style_image.png",output)
