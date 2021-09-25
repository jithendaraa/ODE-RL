#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='channelnorm_cuda',
    ext_modules=[
        CUDAExtension('channelnorm_cuda', [
            'channelnorm_cuda.cc',
            'channelnorm_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
