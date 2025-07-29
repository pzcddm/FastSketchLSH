from setuptools import setup, Extension
import pybind11
import sys

ext_module = Extension(
    'FastSketchLSH',
    sources=[
        'cpp/cminhash.cpp',
        "cpp/kminhash.cpp",
        'cpp/murmurhash3.cpp',
        'cpp/init.cpp'
    ],
    include_dirs=['include', pybind11.get_include()],
    language='c++',
    extra_compile_args=[
        '-std=c++11', '-O3', '-march=native', '-fvisibility=hidden'
    ],
)

setup(
    name='FastSketchLSH',
    version='0.2.0',
    description='High-performance MinHash implementation in C++',
    ext_modules=[ext_module],
    license='MIT',
    python_requires='>=3.7',
    install_requires=['pybind11>=2.6'],
)