from setuptools import setup, Extension
import sys
import platform

# Try to import pybind11, but don't fail if it's not available during setup
try:
    import pybind11
    pybind11_available = True
except ImportError:
    pybind11_available = False

# Cross-platform compile/link options
compile_args = []
link_args = []
define_macros = [
    ('USE_AVX2', '1'),
    ('PYBIND11_STRICT_ASSERTS', '1'),
]

system_name = platform.system()

if system_name == "Windows":
    # MSVC flags
    compile_args = ["/arch:AVX2", "/std:c++17", "/fp:fast", "/Oi"]
elif system_name == "Darwin":
    # macOS uses libc++ (do not try to static link libstdc++)
    compile_args = [
        "-mavx2",
        "-mbmi2",
        "-std=c++17",
        "-ffast-math",
        "-fvisibility=hidden",
        "-stdlib=libc++",
        "-O3",
    ]
    link_args.extend(["-stdlib=libc++"])
else:  # Linux and others with libstdc++
    # Require AVX-512F/DQ/VL to use 512-bit instructions and 64-bit integer ops in fasthash_simd.cpp
    compile_args = [
        "-mavx2",
        "-mbmi2",
        "-mavx512f",
        "-mavx512dq",
        "-mavx512vl",
        "-std=c++17",
        "-ffast-math",
        "-fvisibility=hidden",
        "-O3",
    ]

    # Prefer old C++11 dual ABI for broad compatibility with older libstdc++
    # This avoids requiring newer GLIBCXX_* symbols at runtime on target machines
    define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))

    # Statically link libstdc++/libgcc into the extension so target machines
    # do not need matching libstdc++ versions installed
    link_args.extend(["-static-libstdc++", "-static-libgcc"])

ext_modules = []
if pybind11_available:
    ext_modules.append(Extension(
        'FastSketchLSH',
        sources=[
            'cpp/cminhash.cpp',
            "cpp/kminhash.cpp",
            "cpp/rminhash.cpp",
            "cpp/fasthash_simd.cpp",
            'cpp/murmurhash3.cpp',
            'cpp/fastsketch_lsh.cpp',
            'cpp/init.cpp'
        ],
        include_dirs=['include', pybind11.get_include()],
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=define_macros,
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