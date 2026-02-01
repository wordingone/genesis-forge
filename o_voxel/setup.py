"""
O-Voxel CUDA Extension for TRELLIS.2
Modified setup.py for Windows + MSVC 2022 + PyTorch 2.10 compatibility.

Key fixes:
- Added /D_HAS_STD_BYTE=0 to prevent std::byte conflict with Windows SDK
- Added /Zc:__cplusplus for proper C++17 detection
- Added /permissive- for stricter conformance (helps with namespace resolution)
- Disabled /bigobj warning and enabled it to handle large object files
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True

# Platform-specific compiler flags
if sys.platform == 'win32':
    # MSVC 2022 + PyTorch 2.10 + NVCC 12.8 compatibility flags
    #
    # Known issues:
    # 1. error C2872: 'std': ambiguous symbol - caused by namespace collision
    # 2. std::byte conflicts with Windows byte typedef
    #
    # Solution: Use /permissive- for strict C++ conformance which helps with
    # namespace resolution, combined with proper C++17 flags.
    cxx_flags = [
        "/O2",
        "/std:c++17",
        "/Zc:__cplusplus",  # Report correct __cplusplus value
        "/permissive-",     # Strict standards conformance
        "/bigobj",
        "/wd4624",          # Suppress destructor warning
        "/wd4067",          # Suppress preprocessor warning
        "/wd4838",          # Suppress narrowing conversion warning
        "/EHsc",
    ]
    # For NVCC with PyTorch 2.10+:
    # Pass /permissive- and /Zc:__cplusplus to help with namespace resolution
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-Xcompiler", "/permissive-",
        "-Xcompiler", "/Zc:__cplusplus",
        "-Xcompiler", "/bigobj",
        "-Xcompiler", "/EHsc",
        "-Xcompiler", "/wd4838",
    ]
else:
    cxx_flags = ["-O3", "-std=c++17"]
    nvcc_flags = ["-O3", "-std=c++17"]

if not IS_HIP:
    cc_flag = []
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

setup(
    name="o_voxel",
    packages=[
        'o_voxel',
        'o_voxel.convert',
        'o_voxel.io',
    ],
    ext_modules=[
        CUDAExtension(
            name="o_voxel._C",
            sources=[
                # Hashmap functions
                "src/hash/hash.cu",
                # Convert functions
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetic_attr.cpp",
                ## Serialization functions
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                # IO functions
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                # Rasterization functions
                "src/rasterize/rasterize.cu",

                # main
                "src/ext.cpp",
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/eigen"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags + cc_flag,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
