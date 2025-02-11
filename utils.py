import pyfeats
from typing import *
import numpy as np

class LowLevelFeatureExtractor:
    def __init__(self, function: Callable, # pyfeats function to call
                 params: Dict[str, Any] = None, # pyfeats function parameters
                 features_set: List[str] = None, # list of features to extract
                 ) -> None:
        self.function = function
        self.params = params if params is not None else {}
        self.features_set = features_set

    def __call__(self, image):
        self.params['f'] = image

        features_output = self.function(**self.params)
        
        features_set = {feature: value for feature, value in zip(self.features_set, features_output)}

        return features_set

