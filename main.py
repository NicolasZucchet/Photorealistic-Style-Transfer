import copy
from args import parse_args
import datetime
import time
import logging
import sys

from models import get_model_and_losses
from toolbox import get_experiment_parameters, configure_logger, get_optimizer, get_experiment, save_all, save_images, generate_plots, emptyLogger
from metrics import get_listener

from toolbox.image_preprocessing import plt_images

def create_experience(query = None, parameters = None):
    if parameters is None:
        if query is None:
            query = sys.argv[1:]
        else:
            query = query.split(" ")[1:]
        args = parse_args(prog=query)
    
        parameters = get_experiment_parameters(args)

    parameters.disp()

    if not(parameters.ghost):
        configure_logger(parameters.res_dir+"experiment.log")
        log = logging.getLogger("main")
    else:
        log = emptyLogger()

    experiment = get_experiment(parameters)


    listener = get_listener(parameters.no_metrics,parameters.resume_model,parameters.load_listener_path)
    log.info("experiment and listener objects created")

    model, losses =  get_model_and_losses(experiment, parameters, experiment.content_image)
    log.info("model and losses objects created")

    optimizer, scheduler = get_optimizer(experiment, parameters, losses)
    log.info("optimizer and scheduler objects created")

    log.info('Experiment ' + parameters.save_name+ ' started on {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    if parameters.resume_model:
        log.info('Experiment was recovered from {}'.format(parameters.load_name))

    return {"parameters":parameters, "log":log, "experiment":experiment, "listener":listener, "model":model, "losses":losses, "optimizer":optimizer, "scheduler":scheduler}


def run_experience(experiment, model, parameters, losses, optimizer, scheduler, listener, log):

    # initialize the losses with a forward pass
    experiment.input_image.data.clamp_(0, 1)
    optimizer.zero_grad()
    model.forward(experiment.input_image)

    while experiment.local_epoch < parameters.num_epochs :

        def closure():
            """
            https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
            """
            # meta 
            start_time = time.time()
            meters = listener.reset_meters("train")

            # init
            experiment.input_image.data.clamp_(0, 1)
            optimizer.zero_grad()
            model.forward(experiment.input_image)
            

            style_loss = losses.compute_style_loss()
            # total_loss = style_loss
            meters["style_loss"].update(style_loss.item())

            content_loss = losses.compute_content_loss()
            # total_loss += content_loss
            meters["content_loss"].update(content_loss.item())

            (content_loss+style_loss).backward(retain_graph = True)

            if parameters.reg:
                reg_loss = losses.compute_reg_loss(experiment.input_image)
                meters["reg_loss"].update(reg_loss.item())

            meters["total_loss"].update(reg_loss.item()+style_loss.item()+ reg_loss.item() if parameters.reg else 0)

            if parameters.verbose:
                    print(
                    "\repoch {}:".format(experiment.epoch),
                    "S: {:.5f} C: {:.5f} R: {:.5f}".format(
                        style_loss.item(), content_loss.item(), reg_loss.item() if parameters.reg else 0
                        ),
                    end = "")

            scheduler.step()
            meters["lr"].update(optimizer.state_dict()['param_groups'][0]['lr'])
            experiment.local_epoch += 1
            experiment.epoch += 1

            meters["epoch_time"].update(time.time()-start_time)        
            listener.log_meters("train",experiment.epoch)
            
            return reg_loss + style_loss + reg_loss if parameters.reg else 0
        
        optimizer.step(closure)

def main():

    experience = create_experience()

    parameters = experience["parameters"]
    experiment = experience["experiment"]
    listener = experience["listener"]
    log = experience["log"]
    optimizer = experience["optimizer"]
    losses = experience["losses"]
    model = experience["model"] 
    scheduler = experience["scheduler"]

    run_experience(experiment, model, parameters, losses, optimizer, scheduler, listener, log)

    print()

    experiment.input_image.data.clamp_(0, 1)

    log.info("Done style transfering over "+str(experiment.epoch)+" epochs!")
    

    if parameters.save_model:
        save_all(experiment,model,parameters,listener)
    if not(parameters.ghost):
        save_images(parameters.res_dir+"output.png",experiment.style_image,experiment.input_image,experiment.content_image)
    plt_images(experiment.style_image,experiment.input_image,experiment.content_image)

    if not(parameters.no_metrics):
        generate_plots(parameters, listener)

    print("All done")





if __name__=="__main__":
    main()

