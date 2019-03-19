import os
import sys
import argparse
import shutil

def parse_args(prog = sys.argv[1:]):
    parser = argparse.ArgumentParser(description='')

    
    # name of the experiment
    parser.add_argument('-name', default='', type=str, help='name of experiment. Can only be ommited if you resume an experiment or are running a "-quick" experiment')
   
    parser.add_argument('-quick', default=False, action='store_true', help='run a very quick test to verify a new pipeline, this will overwrite many parameters. If no name is given, default one will be "quick test"')
    # display
    parser.add_argument('-quiet', default=False, action='store_true', help='Do not display the learning process in a verbose way')
    # saving
    parser.add_argument('-no_save', default=False, action='store_true', help='do not save the model at the end of the experiment')
    # disable all optional parameters
    parser.add_argument('-ghost', default=False, action='store_true', help='disable all optional parameters')
    # plotting
    parser.add_argument('-no_metrics', default=False, action='store_true', help='do not records metrics for this experiment')
    # images to use
    parser.add_argument('-style_image', default='1', type=str, help='name of the style image to use in the examples directory')
    parser.add_argument('-content_image', default='1', type=str, help='name of the content image to use in the examples directory')
    parser.add_argument('-input_image', default='content', type=str, help='Which image should be used as initial input image : content, style, white, noise')
    parser.add_argument('-imsize', default=512, type=int, help='size to which the images should be resized (on cpu, default will be 32)')



    # resuming
    parser.add_argument('-keep_params', default=False, action='store_true', help='overwrite parameters given by those of the resumed experiment')

    # model settings
    parser.add_argument('-base_model', default='vgg19', type=str,
                        help='base model to be used for feature extraction')
    parser.add_argument('-device', default='cuda', type=str,
                        help='Which device to use : cuda or cpu')
    parser.add_argument('-content_layers', nargs = "+", default=['4_2'],
                        help='select the convolution layers for which we will compute the content losses')
    parser.add_argument('-style_layers', nargs = "+", default=['1_1','2_1','3_1','4_1'],
                        help='select the convolution layers for which we will compute the style losses')
    parser.add_argument('-num_epochs', default=int(2e2), type=int,
                        help='the number of epochs for this train')
    parser.add_argument('-style_weight', default=1e6, type=float,
                        help='the weight given to the style loss')
    parser.add_argument('-content_weight', default=1e2, type=float,
                        help='the weight given to the content loss')
    parser.add_argument('-reg_weight', default=1, type=float,
                        help='the weight given to the regularization loss')

    # optimizer settings
    parser.add_argument('-optimizer', default="rmsprop", type=str,
                        help='the optimizer that should be used (adam, sgd, lbfgs, rmsprop')
    parser.add_argument('-lr', default=1e-2, type=float,
                        help='the learning rate for the optimizer')
    parser.add_argument('-momentum', default=0.2, type=float,
                        help='the optimizer momentum (used only for adam and sgd')
    parser.add_argument('-weight_decay', default=1e-3, type=float,
                        help='the optimizer weight decay (used only for adam)')

    # scheduler settings
    parser.add_argument('-scheduler', default="step", type=str,
                        help='the type of lr scheduler used (step,exponential,plateau) ')
    parser.add_argument('-lr_step', default=int(100), type=int,
                        help='the epoch step between learning rate drops (for StepScheduler and Plateau)')
    parser.add_argument('-lr_decay', default=int(1e-1), type=float,
                        help='the lr decay momentum/gamma (used for step and exponential decay)')
    # misc settings
    parser.add_argument('-seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    args = parser.parse_args(args=prog)

    # update args

    if args.ghost:
        args.no_save = True
        args.no_log = True
        args.no_metrics = True
    else:
        args.no_log = False

    args.save_model = not(args.no_save)
    args.__delattr__("no_save")

    size = args.imsize
    args.imsize = (size,size) if args.device == "cuda" else (32,32)

    args.verbose = not(args.quiet)
    args.__delattr__("quiet")

    ## LISTS

    args.style_layers = ["conv"+el for el in args.style_layers]
    args.content_layers = ["conv"+el for el in args.content_layers]


    ## PATHS AND NAMES
    
    args.work_dir = ""
    args.save_name = args.load_name if args.name == "" else args.name 

    if args.quick:
        args.num_epochs = 1
        args.content_layers = ["conv0_1"]
        args.style_layers = ["conv0_2"]
        # args.base_model = "quick"
        if args.save_name == "":
            args.save_name = "quick test"

    args.name = args.save_name

    if args.name == "" and not(args.ghost):
        raise Exception("You must enter a name for the experiment (-name) or specify that it is a quick experiment (-quick)")

    args.res_dir = '{}experiments/{}/'.format(args.work_dir, args.save_name)
    args.save_parameters_path = '{}experiments/{}/save/parameters.json'.format(args.work_dir, args.save_name)
    args.save_model_path = '{}experiments/{}/save/model.pt'.format(args.work_dir, args.save_name)
    args.save_experiment_path = '{}experiments/{}/save/experiment.dat'.format(args.work_dir, args.save_name)
    args.save_listener_path = '{}experiments/{}/save/listener.json'.format(args.work_dir, args.save_name)
    # TMP SAVE PATHS NOT YET IMPLEMENTED

    args.style_image_path = '{}examples/{}.png'.format(args.work_dir,args.style_image)
    args.content_image_path = '{}examples/{}.png'.format(args.work_dir,args.content_image)
    args.seg_style_path = '{}examples/{}_seg.png'.format(args.work_dir,args.style_image)
    args.seg_content_path = '{}examples/{}_seg.png'.format(args.work_dir,args.content_image)

    if not(args.ghost) and os.path.exists('{}experiments/{}'.format(args.work_dir, args.save_name)):
        cont = input("You have entered an experiment name that already exists even though you are not resuming that experiment, do you wish to continue (this will delete the folder: "+args.res_dir+"). [y/n] ") == "y"
        if cont:
            shutil.rmtree(args.res_dir)
        else:
            sys.exit(0)
    
    # os.makedirs(args.tmp_dir+"save/",exist_ok=True) # is recursive
    if not(args.ghost):
        os.makedirs(args.res_dir+"save/",exist_ok=True) # is recursive

    assert args.res_dir is not None

    return args
