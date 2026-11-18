import glob
import os
import shutil
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
from setuptools import Extension


pyimmer = Extension(
    name="pyimmer",
    define_macros=[],
    sources=glob.glob("src/extensions/pyimmer/*.cpp"),
    include_dirs=["vendor/immer/"],
    libraries=[],
    library_dirs=[],
    extra_compile_args=["/O2", "/std:c++20"] if os.name == "nt" else ["-O3", "-std=c++2a"],
)

ext_modules = [pyimmer]


def build():
    # The experimental extension is not used by composit itself.
    enabled = os.environ.get("COMPOSIT_BUILD_PYIMMER", "0" if os.name == "nt" else "1")
    if enabled == "0":
        return
    if not os.path.isdir("vendor/immer/immer"):
        raise RuntimeError("Fetch vendor/immer first; see README, or set COMPOSIT_BUILD_PYIMMER=0")

    distribution = Distribution({"name": "extended", "ext_modules": ext_modules})
    distribution.package_dir = "extended"

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
