"""
@File: setup.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: For building the pytorch cuda skeleton, speeded up version compared with DDC
"""

import glob
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# if there is more then appen more
include_dirs = [
    os.path.join(ROOT_DIR, "include"),
    os.path.join(ROOT_DIR, "third_party")
]

sources = glob.glob('*.cpp') + glob.glob('*.cu')

setup(
    name='woot_cuda_skeleton_fin',
    version='0.1',
    author='hemingzhu',
    description='woot_cuda_skeleton_fin',
    long_description='cuda_skeleton',
    ext_modules=[
        CUDAExtension(
            name='woot_cuda_skeleton_fin',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-std=c++14','-O3', '-pthread', '-mavx2', '-mfma', '-fopenmp'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
