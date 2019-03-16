from metrics.metrics_listener import init_listener

def get_listener(no_metrics = False, path = None):
    return init_listener(no_metrics=no_metrics,path=path)