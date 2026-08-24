from .HardNegativeNLLLoss import HardNegativeNLLLoss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("enter loss.utils1")
def load_loss(loss_class, *args, **kwargs):
    print("enter loss.utils2", flush=True)
    if loss_class == "HardNegativeNLLLoss":
        loss_cls = HardNegativeNLLLoss
        logger.info("enter loss.utils2")
    else:
        raise ValueError(f"Unknown loss class {loss_class}")
    return loss_cls(*args, **kwargs)
