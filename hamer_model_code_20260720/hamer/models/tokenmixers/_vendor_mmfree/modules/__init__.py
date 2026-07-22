# hamer/hamer/models/tokenmixers/_vendor_mmfree/modules/__init__.py

# Required by hgrn_bit / convolution
from .fused_norm_gate import FusedRMSNormSwishGate
from .convolution import ShortConvolution
from .layernorm import RMSNorm

# NEW: export ACT2FN (convolution.py imports it as from mmfreelm.modules.activations import ACT2FN)
from .activations import ACT2FN

# If upstream uses others you copied, export them here:
# from .activations import swiglu
