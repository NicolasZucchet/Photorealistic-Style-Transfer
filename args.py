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
    parser.add_argument('-verbose', default=False, action='store_true', help='display the learning process in a verbose way')
    # saving
    parser.add_argument('-no_save', default=False, action='store_true', help='do not save the model at the end of the experiment')
    # plotting
    parser.add_argument('-no_metrics', default=False, action='store_true', help='do not records metrics for this experiment')
    # images to use
    parser.add_argument('-style_image', default='1', type=str, help='ID of the style image to use')
    parser.add_argument('-content_image', default='1', type=str, help='ID of the content image to use')



    # resuming
    parser.add_argument('-resume', default='', type=str, metavar='PATH',
                        help='resume from a given model by providing its name (can be found in the log) ')
    parser.add_argument('-keep_params', default=False, action='store_true', help='overwrite parameters given by those of the resumed experiment')

    # model settings
    parser.add_argument('-no_reg', default=False, action='store_true', 
                        help='Disable regularization')
    parser.add_argument('-base_model', default='vgg19', type=str,
                        help='base model to be used for feature extraction')
    parser.add_argument('-device', default='cuda', type=str,
                        help='Which device to use : cuda or cpu')
    parser.add_argument('-content_layers', nargs = "+", default=['4_2'],
                        help='select the convolution layers for which we will compute the content losses')
    parser.add_argument('-style_layers', default=['1_1','2_1','3_1','4_1'],
                        help='select the convolution layers for which we will compute the style losses')
    parser.add_argument('-num_epochs', default=int(2e2), type=int,
                        help='the number of epochs for this train')
    parser.add_argument('-style_weight', default=1e6, type=float,
                        help='the weight given to the style loss')
    parser.add_argument('-content_weight', default=1e4, type=float,
                        help='the weight given to the content loss')
    parser.add_argument('-reg_weight', default=1e-2, type=float,
                        help='the weight given to the regularization loss')

    # optimizer settings
    parser.add_argument('-optimizer', default="lbfgs", type=str,
                        help='the optimizer that should be used (adam, sgd, lbfgs')
    parser.add_argument('-lr', default=int(1), type=float,
                        help='the learning rate for the optimizer')
    parser.add_argument('-momentum', default=int(0.9), type=float,
                        help='the optimizer momentum (used only for adam and sgd')
    parser.add_argument('-weight_decay', default=int(1e-3), type=float,
                        help='the optimizer weight decay (used only for adam)')

    # scheduler settings
    parser.add_argument('-scheduler', default="plateau", type=str,
                        help='the type of lr scheduler used (step,exponential,plateau) ')
    parser.add_argument('-lr_step', default=int(5e2), type=int,
                        help='the epoch step between learning rate drops (for StepScheduler and Plateau)')
    parser.add_argument('-lr_decay', default=int(1e-1), type=float,
                        help='the lr decay momentum/gamma (used for step and exponential decay)')
    # misc settings
    parser.add_argument('-seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    args = parser.parse_args(args=prog)

    # update args
    
    args.reg = not(args.no_reg)
    args.__delattr__("no_reg")

    args.save_model = not(args.no_save)
    args.__delattr__("no_save")

    args.resume_model = args.resume != ""

    if args.resume_model:
        args.load_name = args.resume
    else:
        args.load_name = ""
    
    args.__delattr__("resume")

    ## LISTS

    args.style_layers = ["conv"+el for el in args.style_layers]
    args.content_layers = ["conv"+el for el in args.content_layers]


    ## PATHS AND NAMES
    
    args.work_dir = ""
    args.save_name = args.load_name if args.name == "" else args.name 

    if args.quick:
        args.num_epochs = 1
        args.base_model = "quick"
        if args.save_name == "":
            args.save_name = "quick test"

    args.name = args.save_name

    if args.name == "":
        raise Exception("You must enter a name for the experiment (-name) or specify that it is a quick experiment (-quick)")

    args.res_dir = '{}experiments/{}/'.format(args.work_dir, args.save_name)
    # args.tmp_dir = '{}experiments/{}/'.format(args.work_dir,"tmp")
    args.load_path = '{}experiments/{}/save/'.format(args.work_dir, args.load_name)
    args.load_model_path = '{}experiments/{}/save/model.pt'.format(args.work_dir, args.load_name)
    args.load_parameters_path = '{}experiments/{}/save/parameters.json'.format(args.work_dir, args.load_name)
    args.load_experiment_path = '{}experiments/{}/save/experiment.dat'.format(args.work_dir, args.load_name)
    args.load_listener_path = '{}experiments/{}/save/listener.json'.format(args.work_dir, args.save_name)
    args.save_parameters_path = '{}experiments/{}/save/parameters.json'.format(args.work_dir, args.save_name)
    args.save_model_path = '{}experiments/{}/save/model.pt'.format(args.work_dir, args.save_name)
    args.save_experiment_path = '{}experiments/{}/save/experiment.dat'.format(args.work_dir, args.save_name)
    args.save_listener_path = '{}experiments/{}/save/listener.json'.format(args.work_dir, args.save_name)
    # TMP SAVE PATHS NOT YET IMPLEMENTED

    args.style_image_path = '{}examples/style/tar{}.png'.format(args.work_dir,args.style_image)
    args.content_image_path = '{}examples/input/in{}.png'.format(args.work_dir,args.content_image)
    args.seg_style_path = '{}examples/segmentation/tar{}.png'.format(args.work_dir,args.style_image)
    args.seg_content_path = '{}examples/segmentation/in{}.png'.format(args.work_dir,args.content_image)

    if args.resume_model and not(os.path.exists('{}experiments/{}/'.format(args.work_dir, args.load_name))):
        raise Exception("Tried to retrieve a model that does not exist at location : "+'{}experiments/{}/'.format(args.work_dir, args.load_name))
    
    if os.path.exists('{}experiments/{}'.format(args.work_dir, args.save_name)) and (args.load_name!=args.save_name or not(args.resume_model)):
        cont = input("You have entered an experiment name that already exists even though you are not resuming that experiment, do you wish to continue (this will delete the folder: "+args.res_dir+"). [y/n] ") == "y"
        if cont:
            shutil.rmtree(args.res_dir)
        else:
            sys.exit(0)
    
    if os.path.exists(args.tmp_dir):
        shutil.rmtree(args.tmp_dir)

    # os.makedirs(args.tmp_dir+"save/",exist_ok=True) # is recursive
    os.makedirs(args.res_dir+"save/",exist_ok=True) # is recursive

    assert args.res_dir is not None

    return args