#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='resample2d_cuda',
    ext_modules=[
        CUDAExtension('resample2d_cuda', [
            'resample2d_cuda.cc',
            'resample2d_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
