"""Backend abstraction layer for NumPy/CuPy compatibility

This module provides a unified interface for array operations that works
with both NumPy (CPU) and CuPy (GPU). The backend is selected via the
USE_GPU environment variable or through configuration.

Environment Variables:
    GAWAIN_USE_GPU: Set to "1" to enable GPU acceleration (default: "0")

Examples:
    >>> # Use NumPy backend (default)
    >>> from gawain.backend import xp
    >>> arr = xp.ones((10, 10))

    >>> # Use CuPy backend
    >>> import os
    >>> os.environ["GAWAIN_USE_GPU"] = "1"
    >>> from gawain.backend import xp
    >>> arr = xp.ones((10, 10))  # This will be on GPU
"""

import os
import sys
from typing import Any

# Determine backend from environment variable or default to NumPy
USE_GPU = os.environ.get("GAWAIN_USE_GPU", "0") == "1"

if USE_GPU:
    try:
        import cupy as xp

        BACKEND = "cupy"
        print("Using CuPy (GPU) backend", file=sys.stderr)
    except ImportError:
        import numpy as xp

        BACKEND = "numpy"
        print(
            "Warning: CuPy not available, falling back to NumPy (CPU) backend",
            file=sys.stderr,
        )
else:
    import numpy as xp

    BACKEND = "numpy"


def to_cpu(array: Any) -> Any:
    """Convert array to CPU (NumPy) if needed

    This function handles the transfer of data from GPU to CPU memory.
    If the array is already on CPU (NumPy), it returns the array unchanged.

    Parameters
    ----------
    array : array-like
        Input array that may be on GPU or CPU

    Returns
    -------
    numpy.ndarray
        Array guaranteed to be on CPU (NumPy array)

    Examples
    --------
    >>> import numpy as np
    >>> cpu_array = np.ones((5, 5))
    >>> result = to_cpu(cpu_array)  # Returns unchanged
    >>> assert isinstance(result, np.ndarray)
    """
    if BACKEND == "cupy":
        return xp.asnumpy(array)
    return array


def to_gpu(array: Any) -> Any:
    """Convert array to GPU (CuPy) if GPU backend is enabled

    This function handles the transfer of data from CPU to GPU memory.
    If GPU backend is not enabled, it returns the array unchanged.

    Parameters
    ----------
    array : array-like
        Input array to be transferred to GPU

    Returns
    -------
    array-like
        Array on GPU if GPU backend is enabled, otherwise unchanged

    Examples
    --------
    >>> import numpy as np
    >>> cpu_array = np.ones((5, 5))
    >>> gpu_array = to_gpu(cpu_array)
    """
    if BACKEND == "cupy":
        return xp.asarray(array)
    return array


def synchronize() -> None:
    """Synchronize GPU operations

    This function ensures all GPU operations have completed before
    proceeding. This is important before timing operations or when
    CPU needs to access GPU data.

    Examples
    --------
    >>> from gawain.backend import xp, synchronize
    >>> arr = xp.ones((1000, 1000))
    >>> result = arr @ arr  # GPU operation
    >>> synchronize()  # Wait for GPU to finish
    """
    if BACKEND == "cupy":
        xp.cuda.Stream.null.synchronize()


def get_array_module(array: Any):
    """Get the appropriate array module (numpy or cupy) for an array

    Parameters
    ----------
    array : array-like
        Input array

    Returns
    -------
    module
        The numpy or cupy module appropriate for the array

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.ones((5, 5))
    >>> mod = get_array_module(arr)
    >>> assert mod == np
    """
    if BACKEND == "cupy":
        return xp.get_array_module(array)
    return xp


def get_backend_info() -> dict:
    """Get information about the current backend

    Returns
    -------
    dict
        Dictionary containing backend information including:
        - backend: "numpy" or "cupy"
        - device: Device information if using GPU
        - version: Backend version string

    Examples
    --------
    >>> info = get_backend_info()
    >>> print(f"Backend: {info['backend']}")
    """
    info = {"backend": BACKEND, "version": xp.__version__}

    if BACKEND == "cupy":
        try:
            device = xp.cuda.Device()
            info["device"] = device.id
            info["device_name"] = (
                device.name.decode()
                if hasattr(device.name, "decode")
                else str(device.name)
            )
            info["memory_pool"] = xp.get_default_memory_pool().used_bytes() / (1024**3)
        except Exception:
            info["device"] = "Unknown"
            info["device_name"] = "Unknown"

    return info
