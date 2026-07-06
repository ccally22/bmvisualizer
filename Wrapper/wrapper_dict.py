from Wrapper.ANNY_WRAPPER import ANNY_WRAPPER
from Wrapper.STAR_WRAPPER import STAR_WRAPPER
from Wrapper.SUPR_WRAPPER import SUPR_WRAPPER

WRAPPER_CLASSES = {
    'SUPR': SUPR_WRAPPER,
    'ANNY': ANNY_WRAPPER,
    'STAR': STAR_WRAPPER,
}

try:
    from Wrapper.MHR_WRAPPER import MHR_WRAPPER
except ModuleNotFoundError:
    MHR_WRAPPER = None
else:
    WRAPPER_CLASSES['MHR'] = MHR_WRAPPER
