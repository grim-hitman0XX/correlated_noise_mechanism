from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.ddp_perlayeroptimizer import (
    DistributedPerLayerOptimizer,
    SimpleDistributedPerLayerOptimizer,
)
from opacus.optimizers.ddpoptimizer import DistributedDPOptimizer
from opacus.optimizers.ddpoptimizer_fast_gradient_clipping import (
    DistributedDPOptimizerFastGradientClipping,
)
from .optimizer import CNMOptimizer
from opacus.optimizers.optimizer_fast_gradient_clipping import (
    DPOptimizerFastGradientClipping,
)
from opacus.optimizers.perlayeroptimizer import DPPerLayerOptimizer


__all__ = [
    "AdaClipDPOptimizer",
    "DistributedPerLayerOptimizer",
    "DistributedDPOptimizer",
    "DPOptimizer",
    "DPOptimizerFastGradientClipping",
    "DistributedDPOptimizerFastGradientlipping",
    "DPPerLayerOptimizer",
    "SimpleDistributedPerLayerOptimizer",
]


def get_optimizer_class(clipping: str, distributed: bool, grad_sample_mode: str = None):
    if grad_sample_mode == "ghost":
        if clipping == "flat" and distributed is False:
            return DPOptimizerFastGradientClipping
        elif clipping == "flat" and distributed is True:
            return DistributedDPOptimizerFastGradientClipping
        else:
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
            )
    elif clipping == "flat" and distributed is False:
        return CNMOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        if grad_sample_mode == "hooks":
            return DistributedPerLayerOptimizer
        elif grad_sample_mode == "ew":
            return SimpleDistributedPerLayerOptimizer
        else:
            raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    elif clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer
    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
