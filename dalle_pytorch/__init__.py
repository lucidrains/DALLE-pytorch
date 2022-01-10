from dalle_pytorch.dalle_pytorch import DALLE, CLIP, DiscreteVAE
from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE

from pkg_resources import get_distribution
__version__ = get_distribution('dalle_pytorch').version
