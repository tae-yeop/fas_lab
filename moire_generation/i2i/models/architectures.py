

from .uhdm.denoiser import UHDMDenoiser
from .wavegan.lofgan import Generator
from .moiredet.extractor import MoireDetExtractor
from .ittr.model import ITTR

arch_dict = {'uhdmdenoiser': UHDMDenoiser,
             'wavegan': Generator,
             'moiredet': MoireDetExtractor,
             'ittr': ITTR}