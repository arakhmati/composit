import glob
from setuptools import setup, find_packages, Extension


pyimmer = Extension(
    name="pyimmer",
    define_macros=[],
    include_dirs=["/usr/local/include", "vendor/immer/"],
    libraries=[],
    library_dirs=["/usr/local/lib"],
    sources=glob.glob("pyimmer/*.cpp"),
    extra_compile_args=["-Ofast", "-std=c++2a"],
)

setup(
    name="persistent_numpy",
    version="0.0.0",
    description="Persistent Implementation of NumPy.",
    packages=find_packages(),
    ext_modules=[pyimmer],
)
