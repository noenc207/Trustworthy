from enum import StrEnum


class XAIAlgorithm(StrEnum):
    GRADCAM = "gradcam"
    GRADCAM_PP = "gradcam++"
    SCORECAM = "scorecam"
    LAYERCAM = "layercam"
    HIRESCAM = "hirescam"
    EIGENCAM = "eigencam"
    EIGENGRADCAM = "eigengradcam"
    XGRADCAM = "xgradcam"
    FULLGRAD = "fullgrad"
    GUIDED_BACKPROP = "guided_backprop"
    GUIDED_GRADCAM = "guided_gradcam"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    ABLATIONCAM = "ablationcam"
    DEEPLIFT = "deeplift"
    LRP = "lrp"
    INPUT_X_GRADIENT = "input_x_gradient"
    OCCLUSION = "occlusion"
    FEATURE_ABLATION = "feature_ablation"

class PerturbationType(StrEnum):
    GAUSSIAN_NOISE = "gaussian_noise"
    BLUR = "blur"
    JPEG = "jpeg"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    ROTATION = "rotation"
    TRANSLATION = "translation"
    SCALING = "scaling"
    HORIZONTAL_FLIP = "horizontal_flip"

class SanityTestType(StrEnum):
    WEIGHT_RANDOMIZATION = "weight_randomization"
    LABEL_RANDOMIZATION = "label_randomization"

class ClinicalRelevance(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNACCEPTABLE = "unacceptable"
