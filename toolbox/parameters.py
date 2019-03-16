import json

class Experiment_parameters():

    def __init__(self,args):
        for k,v in args.__dict__.items():
            self.__setattr__(k,v)
    
    def disp(self):
        print('----- Experiments Parameters -----')
        for k, v in self.__dict__.items():
            if k in []:
                continue
            print(k, ':', v)
    
    def save(self):
        with open(self.save_parameters_path,"w") as f:
            json.dump(self.__dict__,f)
    
    def load(self):
        with open(self.load_parameters_path,"r") as f:
            d = json.load(f)
        for k,v in d.items():
            if k in ["resume","keep_params"] or ("path" in k and (not("seg" in k) or not("image" in k))):
                # the parameters we never want to recover
                continue
            if self.keep_params: 
                # if kee_params, all others are loaded
                self.__setattr__(k,v)
            elif "image" in k or "seg" in k or k in ["base_model","scheduler","no_metrics","reg","seed","imsize"]:
                # the parameters we always want to recover
                self.__setattr__(k,v)