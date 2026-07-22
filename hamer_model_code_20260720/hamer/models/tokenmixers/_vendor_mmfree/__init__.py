# hamer/hamer/models/tokenmixers/_vendor_mmfree/__init__.py
import sys as _sys

# 1) Alias THIS package as "mmfreelm" before importing subpackages
_pkg = _sys.modules[__name__]
_sys.modules.setdefault('mmfreelm', _pkg)

# 2) Now import the vendored subpackages (they may import "mmfreelm.*")
from . import modules, layers, ops, utils  # noqa: F401

# 3) (Optional) also publish explicit submodule names in sys.modules
_sys.modules['mmfreelm.modules'] = modules
_sys.modules['mmfreelm.layers']  = layers
_sys.modules['mmfreelm.ops']     = ops
_sys.modules['mmfreelm.utils']   = utils
