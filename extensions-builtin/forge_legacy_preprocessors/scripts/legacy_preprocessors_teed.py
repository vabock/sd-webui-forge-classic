import numpy as np
from skimage import morphology

from annotator.teed import TEEDDetector
from annotator.util import HWC3
from legacy_preprocessors.preprocessor import resize_image_with_pad
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, supported_preprocessors


class PreprocessorTEED(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name="softedge_teed"
        self.tags = ["SoftEdge"]
        self.slider_1 = PreprocessorParameter(
            label="Safe Steps",
            minimum=0,
            maximum=10,
            value=2,
            step=1,
            visible=True,
        )
        self.model = None

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        img, remove_pad = resize_image_with_pad(input_image, resolution)
        if self.model is None:
            self.model = TEEDDetector()

        result = self.model(img, safe_steps=int(slider_1))
        result = HWC3(result)
        return remove_pad(result)


def get_intensity_mask(image_array, lower_bound, upper_bound):
    mask = image_array[:, :, 0]
    mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
    mask = np.expand_dims(mask, 2).repeat(3, axis=2)
    return mask


def combine_layers(base_layer, top_layer):
    mask = top_layer.astype(bool)
    temp = 1 - (1 - top_layer) * (1 - base_layer)
    result = base_layer * (~mask) + temp * mask
    return result


class PreprocessorAnyline(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name="softedge_anyline"
        self.tags = ["SoftEdge"]
        self.slider_resolution = PreprocessorParameter(
            label="Resolution",
            minimum=64,
            maximum=2048,
            value=1280,
            step=8,
            visible=True,
        )
        self.slider_1 = PreprocessorParameter(
            label="Safe Steps",
            minimum=0,
            maximum=10,
            value=2,
            step=1,
            visible=True,
        )
        # self.preprocessor_deps = ["lineart_standard"]
        self.model = None

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        img, remove_pad = resize_image_with_pad(input_image, resolution)
        if self.model is None:
            self.model = TEEDDetector(mteed=True)

        mteed_result = self.model(img, safe_steps=int(slider_1))
        mteed_result = HWC3(mteed_result)

        lineart_preprocessor = None
        for k, v in supported_preprocessors.items():
            if k.startswith("lineart_standard"):
                lineart_preprocessor = v
                break
        assert lineart_preprocessor is not None
        lineart_result = lineart_preprocessor(img, resolution)
        lineart_result = get_intensity_mask(
            lineart_result, lower_bound=0, upper_bound=1
        )
        cleaned = morphology.remove_small_objects(
            lineart_result.astype(bool), min_size=36, connectivity=1
        )
        lineart_result = lineart_result * cleaned
        final_result = combine_layers(mteed_result, lineart_result)
        return remove_pad(final_result)


add_supported_preprocessor(PreprocessorTEED())
add_supported_preprocessor(PreprocessorAnyline())
