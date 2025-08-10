from setuptools import setup, Extension
import sys
import platform

# Try to import pybind11, but don't fail if it's not available during setup
try:
    import pybind11
    pybind11_available = True
except ImportError:
    pybind11_available = False

# 跨平台编译选项
compile_args = []
if platform.system() == "Windows":
    compile_args = ["/arch:AVX2", "/std:c++17", "/fp:fast", "/Oi"]
else:
    # 需要 AVX-512F/DQ/VL 支持以使用 fasthash_simd.cpp 中的 512 位指令与 64 位整数运算
    compile_args = [
        "-mavx2",
        "-mbmi2",
        "-mavx512f",
        "-mavx512dq",
        "-mavx512vl",
        "-std=c++17",
        "-ffast-math",
        "-fvisibility=hidden",
    ]

ext_modules = []
if pybind11_available:
    ext_modules.append(Extension(
        'FastSketchLSH',
        sources=[
            'cpp/cminhash.cpp',
            "cpp/kminhash.cpp",
            "cpp/rminhash.cpp",
            "cpp/fasthash.cpp",
            "cpp/fasthash_simd.cpp",
            'cpp/murmurhash3.cpp',
            'cpp/init.cpp'
        ],
        include_dirs=['include', pybind11.get_include()],
        language='c++',
        extra_compile_args=compile_args,
        define_macros=[
            ('USE_AVX2', '1'),
            ('PYBIND11_STRICT_ASSERTS', '1')
        ],
    ))

setup(
    name='FastSketchLSH',
    version='0.3.1',
    description='High-performance MinHash with SIMD acceleration',
    ext_modules=ext_modules,
    license='MIT',
    python_requires='>=3.7',
    install_requires=['pybind11>=2.10'],
    setup_requires=['pybind11>=2.10'],
)