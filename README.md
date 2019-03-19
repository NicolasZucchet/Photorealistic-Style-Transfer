# Photorealistic-Style-Transfer

Amaury Sudrie, Victor Ruelle, Nicolas Zucchet

## Objective
Based on the Yagudin et. al's pytorch implementation of the Deep Photo Style Transfer paper ( Luan et. al ), we aim to explore new applications and modifications of photo styletransfer. To perform photo style transfer segmentation is used. The quality of the transfer then depends on the style photo. We are mainly concerned with how to find a good style photo with respect to user style will. Moreover we aim to automate the whole process.

## How to run the code

Every possible expirement parameter can be tuned in the command line using special arguments (if you find one missing, please contact Victor to implement it).

A classical experiment will be launched using : python main.py -name "an experiment name" 

Such a command will automatically create directories for the experiment and save/log everthing in there (the name of the directory will be the name given by the -name flag, more details below). By default, after a run, the resulting network is saved in the "save" sub-directory; this will allow you to continue this experiment anytime.

The full list of optional arguments and their description can be easily retrieved by running : python main.py -h 

The main things to be aware of are :
- To run a minimalistc experiment for testing your pipeline, add the -quick flag. This will set some default parameters to quickly test the entire pipeline.
- To resume an experiment, simply run : python main.py -resume NAME_Of_EXPERIMENT_TO_RESUME. Note that this will load some basic parameters from the resumed experiment but ALL OTHER parameters will be default if not specified. This can be avoided by using the -keep_params flag which will recover all non contradictory parameters from the loaded model.
- Results of an experiment will always be saved in a subfolder of "experiments" whose name will be the one provided by using the -name A_NAME flag. The only exceptions to this rule are : if you resume an experiment and you don't specify a different name with the name flag, then, the loaded model will be overwritten OR if you add the -quick flag
- If you want to test an existing model, you should add the -play flag. Only then will the training phase be followed by a testing phase. You can also add the -test flag to run a playing phase without a training phase (equivalent to -play -M 0)
- To have more details on what is going on, you can use the -verbose flag which will dynamically print the progress of the experiment

## Logging

Every important information is kept in the experiment.log. Check it out if you are want to see exactly what happened. 

You can add experiming.log.info(message) at anytime in any file if you want to set up personalized logging messages. They will appear in the same experiment.log file. 

You can also set up a specific logger by calling my_logger = logging.getLogger(NAME), the messages logged by this logger will then be logged in the same file but with the mention NAME before. 


## Plotting results

By default, some pre-defined variables will be recorded during training and playing phases and plots will be generated in the corresponding subfolder of the experiments folder. This can be turned off with the -no_metrics flag. 

This is done by the experiment.listener object which is an instance of the Listener class in metrics_listener.py . To record and plot new stats, you should simply add a line in the make_meters function (in metrics.py) and update the new field when wanted (look at the closure function in the Experiment class for a concise example of usage).


## Credits
The entire code structure is cloned from : https://github.com/yagudin/PyTorch-deep-photo-styletransfer
