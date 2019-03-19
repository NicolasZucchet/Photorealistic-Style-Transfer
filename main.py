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
from torch.nn.utils import clip_grad_norm

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


    listener = get_listener(parameters.no_metrics)
    log.info("experiment and listener objects created")

    model, losses =  get_model_and_losses(experiment, parameters, experiment.content_image)
    log.info("model and losses objects created")

    optimizer, scheduler = get_optimizer(experiment, parameters)
    log.info("optimizer and scheduler objects created")

    log.info('Experiment ' + parameters.save_name+ ' started on {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    return {"parameters":parameters, "log":log, "experiment":experiment, "listener":listener, "model":model, "losses":losses, "optimizer":optimizer, "scheduler":scheduler}


def run_experience(experiment, model, parameters, losses, optimizer, scheduler, listener, log):

    best_loss = 1e10
    best_input = None

    while experiment.local_epoch < parameters.num_epochs :

        def closure():
            nonlocal experiment
            nonlocal best_input
            nonlocal best_loss

            # meta 
            start_time = time.time()
            meters = listener.reset_meters("train")

            # init
            experiment.input_image.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(experiment.input_image)

            style_loss = losses.compute_style_loss()
            meters["style_loss"].update(style_loss.item())

            content_loss = losses.compute_content_loss()
            meters["content_loss"].update(content_loss.item())

            tv_loss = losses.compute_tv_loss()
            meters["tv_loss"].update(tv_loss.item())

            reg_loss = losses.compute_reg_loss(experiment.input_image)
            meters["reg_loss"].update(reg_loss.item())

            loss = style_loss + content_loss + tv_loss + reg_loss

            # Store the best result for outputing
            if loss < best_loss:
                best_loss = loss
                best_input = experiment.input_image.data.clone()

            experiment.input_image.retain_grad()
            loss.backward()

            meters["total_loss"].update(loss.item())

            if parameters.verbose:
                    print(
                    "\repoch {}:".format(experiment.epoch),
                    "S: {:.5f} C: {:.5f} R: {:.5f} TV: {:.5f}".format(
                        style_loss.item(), content_loss.item(), 0 if reg_loss == 0 else reg_loss.item(), tv_loss
                        ),
                    end = "")

            # Gradient cliping deal with gradient exploding
            clip_grad_norm(model.parameters(), 15.0)

            if parameters.scheduler == "plateau":
                scheduler.step(style_loss.item()+content_loss.item()+reg_loss.item() if parameters.reg else 0)
            else:
                scheduler.step()
            meters["lr"].update(optimizer.state_dict()['param_groups'][0]['lr'])
            experiment.local_epoch += 1
            experiment.epoch += 1

            meters["epoch_time"].update(time.time()-start_time)        
            listener.log_meters("train",experiment.epoch)
            
            return loss
        
        optimizer.step(closure)

    experiment.input_image.data = best_input
    experiment.input_image.data.clamp_(0, 1)

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

