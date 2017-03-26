from util import *
from config import *
from reader import *
from runner import *
from tf_runner import *

try:
    import torch
except ImportError:
    pass
else:
    # torch stuff
    from torch_runner import *
