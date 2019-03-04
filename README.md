# Photorealistic-Style-Transfer

## Objective
Based on the Yagudin et. al's pytorch implementation of the Deep Photo Style Transfer paper ( Luan et. al ), we aim to explore new applications and modifications of photo styletransfer. We are mainly concerned with speeding up the process of styletransfer which is extremely time consuming as of now and with exploring how different styles can be combined onto a single content image.

## Roles
Amaury : setting up a new database of images based on a Google image scrapper

Victor : setting up a framework in which the code can easily be edited, run on specific experiments and monitored for performances

Nicolas : Looking for changes which could improve the time needed to process an image. 

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

The logging is not extensive for now... You can add experiming.log.info(message) at anytime in any file if you want to set up personalized logging messages. They will appear in the same experiment.log file. 

You can also set up a specific logger by calling my_logger = logging.getLogger(NAME), the messages logged by this logger will then be logged in the same file but with the mention NAME before. 


## Plotting results

By default, some pre-defined variables will be recorded during training and playing phases and plots will be generated in the corresponding subfolder of the experiments folder. This can be turned off with the -no_metrics flag. 

This is done by the experiment.listener object which is an instance of the Listener class in metrics_listener.py . To record and plot new stats, you should simply add a line in the make_meters function (in metrics.py) and update the new field when wanted (look at the closure function in the Experiment class for a concise example of usage).

## Changes

CUDA support was added in several functions ( image_preprocessing.image_to_tensor and image_preprocessing.tensor_to_image)

## Problems to be corrected

### 1
When running the script, I noticed that the regularization loss was often negative and was very unstable.
Example :

step  200: S: 89.747 C: 4.231 R:-5559.885

step  250: S: 88.981 C: 5.352 R:1647.456

step  300: S: 89.943 C: 5.703 R:-3204.946

step  350: S: 89.955 C: 13.522 R:-4951.400

step  400: S: 97.520 C: 7.561 R:-3702.983

step  450: S: 229.646 C: 46.452 R:21189.082

step  500: S: 229.500 C: 43.207 R:19783.462

We should investigate into this behavior

### 2
After running 2000 epochs on an image with regularization, I notice that the losses diverge...

step 1150: S: 309.799 C: 82.518 R:52448.856

step 1200: S: 4812.417 C: 4842.881 R:154887.349

step 1250: S: 408.577 C: 193.212 R:54830.851

step 1300: S: 315.698 C: 97.832 R:47640.283

step 1350: S: 35584.047 C: 8521.994 R:392544.276

step 1400: S: 392.760 C: 147.498 R:74679.916

step 1450: S: 325.284 C: 110.247 R:66217.362

step 1500: S: 312.444 C: 75.289 R:63786.109

Earlier in the process we had :

step  250: S: 88.981 C: 5.352 R:1647.456

The resulting image is saved as "problem.png"

## Credits
The entire code structure is cloned from : https://github.com/yagudin/PyTorch-deep-photo-styletransfer
