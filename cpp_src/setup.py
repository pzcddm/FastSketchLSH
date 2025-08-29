from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import platform
import os
import tempfile
import subprocess
import textwrap
import shutil

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
    ('PYBIND11_STRICT_ASSERTS', '1'),
]

# Enable FxHasher prehash if requested via environment variable
if os.getenv('FASTSKETCH_USE_FXHASH') in {"1", "ON", "on", "true", "TRUE", "Yes", "yes"}:
    define_macros.append(('FASTSKETCH_USE_FXHASH', '1'))

system_name = platform.system()

# Baseline flags: no forced AVX so wheels run on older CPUs. We'll detect
# CPU+compiler capabilities in a custom build_ext and append AVX flags.
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
        # Env overrides
        force_avx512 = env_truthy("FASTSKETCH_FORCE_AVX512")
        disable_avx512 = env_truthy("FASTSKETCH_DISABLE_AVX512")
        force_avx2 = env_truthy("FASTSKETCH_FORCE_AVX2")
        disable_avx2 = env_truthy("FASTSKETCH_DISABLE_AVX2")

        is_msvc = self.compiler.compiler_type == "msvc"
        avx512_flags = ["/arch:AVX512"] if is_msvc else ["-mavx512f", "-mavx512dq", "-mavx512vl"]
        avx2_flags = ["/arch:AVX2"] if is_msvc else ["-mavx2", "-mbmi2", "-msse4.2"]

        # CPU detection using CPUID: AVX512F = EBX bit 16, AVX2 = EBX bit 5
        cpu_has_avx512 = self.cpu_has_feature_via_cpuid(16)
        cpu_has_avx2 = self.cpu_has_feature_via_cpuid(5)

        # Compiler support probing
        compiler_has_avx512 = self.has_compiler_flags(avx512_flags)
        compiler_has_avx2 = self.has_compiler_flags(avx2_flags)

        # Decide which flags to add
        use_avx512 = False
        if not disable_avx512 and (force_avx512 or (cpu_has_avx512 and compiler_has_avx512)):
            use_avx512 = True
        use_avx2 = False
        if not use_avx512 and not disable_avx2 and (force_avx2 or (cpu_has_avx2 and compiler_has_avx2)):
            use_avx2 = True

        # Emit informative build-time messages
        print("FastSketchLSH build configuration:")
        print(f"  Compiler: {self.compiler.compiler_type}")
        print(f"  CPU features: AVX512F={'yes' if cpu_has_avx512 else 'no'}, AVX2={'yes' if cpu_has_avx2 else 'no'}")
        print(f"  Compiler flags supported: AVX512={'yes' if compiler_has_avx512 else 'no'}, AVX2={'yes' if compiler_has_avx2 else 'no'}")
        if force_avx512 or force_avx2 or disable_avx512 or disable_avx2:
            print("  Overrides:")
            if force_avx512: print("    FASTSKETCH_FORCE_AVX512=1")
            if disable_avx512: print("    FASTSKETCH_DISABLE_AVX512=1")
            if force_avx2: print("    FASTSKETCH_FORCE_AVX2=1")
            if disable_avx2: print("    FASTSKETCH_DISABLE_AVX2=1")
        if use_avx512:
            print("  Selected SIMD: AVX-512 (F/DQ/VL)")
        elif use_avx2:
            print("  Selected SIMD: AVX2 (+BMI2/SSE4.2 where available)")
        else:
            print("  Selected SIMD: baseline (no AVX-specific instructions)")

        for ext in self.extensions:
            # Append ISA flags to existing base flags
            if use_avx512:
                ext.extra_compile_args += avx512_flags
                # Ensure preprocessor paths for AVX-512-enabled code are visible
                ext.define_macros = list(getattr(ext, 'define_macros', [])) + [("__AVX512F__", "1")]
            elif use_avx2:
                ext.extra_compile_args += avx2_flags
        super().build_extensions()

ext_modules = []
if pybind11_available:
    ext_modules.append(Extension(
        'FastSketchLSH',
        sources=[
            # 'cpp/cminhash.cpp',
            "cpp/rminhash.cpp",
            # scalar fasthash is deprecated; keep file for review only, exclude from wheel
            # "cpp/fasthash.cpp",
            "cpp/fasthash_deprecated.cpp",
            "cpp/fastsketch.cpp",
            'cpp/murmurhash3.cpp',
            'cpp/fastsketch_lsh.cpp',
            'cpp/fastsketch_rensa_lsh.cpp',
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
    cmdclass={'build_ext': BuildExt},
    license='MIT',
    python_requires='>=3.7',
    install_requires=['pybind11>=2.10'],
    setup_requires=['pybind11>=2.10'],
)