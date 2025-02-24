from typing import *
import numpy as np

class LowLevelFeatureExtractor:
    def __init__(self, function: Callable, # pyfeats function to call
                 params: Dict[str, Any] = None, # pyfeats function parameters
                 features_set: List[str] = None, # list of features to extract,
                 image_size: Tuple[int, int] = None # size of the input image (width, height) 
                 ) -> None:
        self.function = function
        self.params = params if params is not None else {}
        self.features_set = features_set
        self.image_size = image_size if image_size is not None else (640, 640)

    def __call__(self, images):
        images = np.array(images) if not isinstance(images, np.ndarray) else images
        features = []

        for image in images:
            image= np.squeeze(image)

            features_output = self.function(image, **self.params)
            
            features_set = {feature: value for feature, value in zip(self.features_set, features_output)}

            features_set = np.concatenate([features_set[key] for key in features_set.keys()], axis=0)

            features.append(features_set)

        return np.stack(features, axis=0)
    
    def process_single_image(self, image:np.ndarray) -> np.ndarray:
        features = self.function(image, **self.params)
        features = {feature: value for feature, value in zip(self.features_set, features)}
        features = np.concatenate([features[key] for key in features.keys()], axis=0)
        return features

    def get_features_size(self) -> int:
        sample_image = np.random.randint(0, 256, self.image_size).astype("uint8")
        features_set = self.process_single_image(sample_image)

        self.features_size = features_set.shape[0]

        return self.features_size