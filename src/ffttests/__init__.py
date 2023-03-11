from pkg_resources import get_distribution, DistributionNotFound

from ffttests.usfft1d import *
from ffttests.usfft2d import *
from ffttests.fftcl import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass