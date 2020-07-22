"""
Some codes from
https://github.com/openai/improved-gan/blob/master/imagenet/utils.py
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import os
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
