from src.modules.explainability.strategies.base import BaseCAMStrategy


class LayerCAMStrategy(BaseCAMStrategy):
    pass

class EigenCAMStrategy(BaseCAMStrategy):
    pass

class GuidedBackpropStrategy(BaseCAMStrategy):
    pass

class GuidedGradCAMStrategy(BaseCAMStrategy):
    pass

class IntegratedGradientsStrategy(BaseCAMStrategy):
    pass

class AblationCAMStrategy(BaseCAMStrategy):
    pass

class DeepLIFTStrategy(BaseCAMStrategy):
    pass

class LRPStrategy(BaseCAMStrategy):
    pass
