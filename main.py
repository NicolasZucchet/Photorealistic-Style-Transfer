import copy
from args import parse_args
import datetime
import time
from toolbox.utils import safe_save
from experiment import Experiment
from toolbox.plotter import generate_plots
from toolbox.image_preprocessing import save_images


def manual_mode(query):
    query = query.split(" ")[1:]
    args = parse_args(prog=query)
    exp = Experiment(args)
    return exp

def main():

    args = parse_args()

    exp = Experiment(args)
    exp.disp()
    

    exp.log.info('Experiment ' +exp.parameters.save_name+ ' started on {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    if exp.parameters.resume_model:
        exp.log.info('Experiment was recovered from {}'.format(exp.parameters.load_name))
    exp.log.info('Expirment parameters: '+''.join([str(k)+" : "+str(v)+"; " for k, v in exp.parameters.__dict__.items()]))

    exp.run()

    exp.log.info("Done style transfering over "+str(exp.epoch)+" epochs!")
    
    exp.save() # does nothing if exp.parameters.save_model = False

    if exp.parameters.save:
        save_images(exp.parameters.res_dir+"output.png",exp.style_image,exp.input_image,exp.content_image)

    if not(exp.parameters.no_metrics):
        generate_plots(exp)

    print("All done")


if __name__=="__main__":
    main()