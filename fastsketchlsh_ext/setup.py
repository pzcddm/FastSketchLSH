from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import platform
import os
import tempfile
import subprocess
import textwrap
import shutil
from pathlib import Path

# Try to import pybind11, but don't fail if it's not available during setup
try:
    import pybind11
    pybind11_available = True
except ImportError:
    pybind11_available = False

# Try to import numpy to access its headers during the build
try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False
    np = None

# Cross-platform compile/link options
compile_args = []
link_args = []
define_macros = [
    ('PYBIND11_STRICT_ASSERTS', '1'),
]

# Enable FxHasher prehash if requested via environment variable
if os.getenv('FASTSKETCH_USE_FXHASH') in {"1", "ON", "on", "true", "TRUE", "Yes", "yes"}:
    define_macros.append(('FASTSKETCH_USE_FXHASH', '1'))

system_name = platform.system()

# Baseline flags: no forced x86 ISA so wheels run on older CPUs. We'll detect
# x86 CPU+compiler capabilities in a custom build_ext and append AVX-512 flags.
if system_name == "Windows":
    compile_args = ["/std:c++17", "/fp:fast", "/Oi"]
elif system_name == "Darwin":
    compile_args = [
        "-std=c++17",
        "-ffast-math",
        "-fvisibility=hidden",
        "-stdlib=libc++",
        "-O3",
    ]
    link_args.extend(["-stdlib=libc++"])
else:  # Linux and others with libstdc++
    compile_args = [
        "-std=c++17",
        "-ffast-math",
        "-fvisibility=hidden",
        "-O3",
    ]
    define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))
    link_args.extend(["-static-libstdc++", "-static-libgcc"])


def env_truthy(name: str) -> bool:
    return os.getenv(name) in {"1", "ON", "on", "true", "TRUE", "Yes", "yes"}


def get_numpy_include_dirs():
    if numpy_available and np is not None:
        return [np.get_include()]
    try:
        import numpy as np_local
        return [np_local.get_include()]
    except ImportError:
        return []


class BuildExt(build_ext):
    def has_compiler_flags(self, flags):
        # Try to compile a minimal program with the given flags
        import distutils.errors
        from distutils.ccompiler import new_compiler
        from distutils.sysconfig import customize_compiler
        tmpdir = tempfile.mkdtemp()
        fname = os.path.join(tmpdir, "test_flags.cpp")
        with open(fname, "w") as f:
            f.write("int main(){return 0;}")
        try:
            compiler = new_compiler(compiler=self.compiler.compiler_type)
            customize_compiler(compiler)
            objects = compiler.compile([fname], output_dir=tmpdir, extra_postargs=flags)
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return False
        shutil.rmtree(tmpdir, ignore_errors=True)
        return True

    def cpu_has_feature_via_cpuid(self, bit_index):
        # Query CPUID leaf 7, subleaf 0, EBX bit bit_index.
        # Returns True if present. Works on x86/x64.
        code = textwrap.dedent(
            r"""
            #if defined(_MSC_VER)
            #include <intrin.h>
            #else
            #include <cpuid.h>
            #endif
            #include <stdio.h>
            int main(){
                unsigned int eax=0, ebx=0, ecx=0, edx=0;
                #if defined(_MSC_VER)
                int regs[4];
                __cpuidex(regs, 7, 0);
                ebx = (unsigned)regs[1];
                #else
                if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 1;
                #endif
                if ( (ebx & (1u<<%d)) != 0u ) return 0; else return 1;
            }
            """ % (bit_index,)
        )
        tmpdir = tempfile.mkdtemp()
        src = os.path.join(tmpdir, "cpuid_check.cpp")
        with open(src, "w") as f:
            f.write(code)
        exe = os.path.join(tmpdir, "cpuid_check")
        if os.name == "nt":
            exe += ".exe"
        try:
            # Compile without any special ISA flags
            objects = self.compiler.compile([src], output_dir=tmpdir, extra_postargs=[])
            self.compiler.link_executable(objects, exe)
            # Run: exit code 0 means feature present
            result = subprocess.run([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except Exception:
            return False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def build_extensions(self):
        # Env overrides (x86 only)
        force_avx512 = env_truthy("FASTSKETCH_FORCE_AVX512")
        disable_avx512 = env_truthy("FASTSKETCH_DISABLE_AVX512")

        is_msvc = self.compiler.compiler_type == "msvc"
        avx512_flags = ["/arch:AVX512"] if is_msvc else ["-mavx512f", "-mavx512dq", "-mavx512vl"]

        # Only probe AVX-512 on x86/x64
        machine = platform.machine().lower()
        is_x86 = any(k in machine for k in ("x86_64", "amd64", "i386", "i686"))
        cpu_has_avx512 = False
        compiler_has_avx512 = False
        if is_x86:
            cpu_has_avx512 = self.cpu_has_feature_via_cpuid(16)
            compiler_has_avx512 = self.has_compiler_flags(avx512_flags)

        # Decide which flags to add
        use_avx512 = False
        if is_x86 and not disable_avx512 and (force_avx512 or (cpu_has_avx512 and compiler_has_avx512)):
            use_avx512 = True

        # Emit informative build-time messages
        print("FastSketchLSH build configuration:")
        print(f"  Compiler: {self.compiler.compiler_type}")
        print(f"  Host arch: {machine}")
        if is_x86:
            print(f"  CPU features: AVX512F={'yes' if cpu_has_avx512 else 'no'}")
            print(f"  Compiler flags supported: AVX512={'yes' if compiler_has_avx512 else 'no'}")
        if force_avx512 or disable_avx512:
            print("  Overrides:")
            if force_avx512: print("    FASTSKETCH_FORCE_AVX512=1")
            if disable_avx512: print("    FASTSKETCH_DISABLE_AVX512=1")
        if use_avx512:
            print("  Selected SIMD: AVX-512 (F/DQ/VL)")
        else:
            print("  Selected SIMD: baseline/NEON (no x86-only flags)")

        for ext in self.extensions:
            numpy_dirs = get_numpy_include_dirs()
            if not numpy_dirs:
                print("  NumPy headers not found; install numpy>=1.21 before building FastSketchLSH")
            ext.include_dirs = list(getattr(ext, 'include_dirs', [])) + numpy_dirs
            # Append ISA flags to existing base flags
            if use_avx512:
                ext.extra_compile_args += avx512_flags
                # Ensure preprocessor paths for AVX-512-enabled code are visible
                ext.define_macros = list(getattr(ext, 'define_macros', [])) + [("__AVX512F__", "1")]

            # Try to enable OpenMP (conditionally)
            enable_openmp = False
            if system_name == "Darwin":
                # Prefer Homebrew libomp
                brew_prefixes = ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
                for pref in brew_prefixes:
                    inc = os.path.join(pref, "include")
                    lib = os.path.join(pref, "lib")
                    if os.path.exists(os.path.join(inc, "omp.h")):
                        ext.include_dirs = list(getattr(ext, 'include_dirs', [])) + [inc]
                        ext.library_dirs = list(getattr(ext, 'library_dirs', [])) + [lib]
                        ext.extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                        ext.extra_link_args += ["-lomp", f"-Wl,-rpath,{lib}"]
                        enable_openmp = True
                        print(f"  OpenMP: using libomp at {pref}")
                        break
                if not enable_openmp:
                    print("  OpenMP: libomp not found; building without OpenMP (batch will be single-thread)")
            else:
                # Linux/others: try standard flags; may fail if toolchain lacks OpenMP
                try_flags = ["-fopenmp"]
                if self.has_compiler_flags(try_flags):
                    ext.extra_compile_args += try_flags
                    ext.extra_link_args += try_flags
                    enable_openmp = True
                    print("  OpenMP: enabled with -fopenmp")
                else:
                    print("  OpenMP: not supported by compiler; building without OpenMP")
        super().build_extensions()

ext_modules = []
if pybind11_available:
    base_include_dirs = ['include', pybind11.get_include()]
    base_include_dirs.extend(get_numpy_include_dirs())
    ext_modules.append(Extension(
        'FastSketchLSH',
        sources=[
            "cpp/rminhash.cpp",
            # scalar fasthash is deprecated; keep file for review only, exclude from wheel
            # "cpp/fasthash.cpp",
            "cpp/fasthash_deprecated.cpp",
            "cpp/fastsketch.cpp",
            'cpp/murmurhash3.cpp',
            'cpp/LSH.cpp',
            'cpp/init.cpp'
        ],
        include_dirs=base_include_dirs,
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=define_macros,
    ))

# Read long description from README.md if available
try:
    long_description = Path("README.md").read_text(encoding="utf-8")
except Exception:
    long_description = ""

setup(
    name='FastSketchLSH',
    version='0.1.0',
    description='High-performance FastSketch with SIMD acceleration to deduplicate large-scale data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='FastSketchLSH Authors',
    url='https://github.com/pzcddm/FastSketchLSH',
    project_urls={
        'Source': 'https://github.com/pzcddm/FastSketchLSH',
        'Issues': 'https://github.com/pzcddm/FastSketchLSH/issues',
    },
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    license='MIT',
    license_files=['LICENSE'],
    python_requires='>=3.7',
    install_requires=['pybind11>=2.10', 'numpy>=1.21'],
    setup_requires=['pybind11>=2.10', 'numpy>=1.21'],
)