"""Base callbacks."""

from collections import defaultdict
from copy import deepcopy

def on_predict_start(predictor):
    """Called when the prediction starts."""
    pass


def on_predict_batch_start(predictor):
    """Called at the start of each prediction batch."""
    pass


def on_predict_batch_end(predictor):
    """Called at the end of each prediction batch."""
    pass


def on_predict_postprocess_end(predictor):
    """Called after the post-processing of the prediction ends."""
    pass


def on_predict_end(predictor):
    """Called when the prediction ends."""
    pass

default_callbacks = {
    # Run in predictor
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
}


def get_default_callbacks():
    """
    Return a copy of the default_callbacks dictionary with lists as default values.

    Returns:
        (defaultdict): A defaultdict with keys from default_callbacks and empty lists as default values.
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """
    Add integration callbacks from various sources to the instance's callbacks.

    Args:
        instance (Trainer, Predictor, Validator, Exporter): An object with a 'callbacks' attribute that is a dictionary
            of callback lists.
    """
    callbacks_list = [{}]

    # Add the callbacks to the callbacks dictionary
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
