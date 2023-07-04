import glob
import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.dist import Distribution
from distutils.extension import Extension


pyimmer = Extension(
    name="pyimmer",
    define_macros=[],
    sources=glob.glob("src/extensions/pyimmer/*.cpp"),
    include_dirs=["/usr/local/include", "vendor/immer/"],
    libraries=[],
    library_dirs=["/usr/local/lib"],
    extra_compile_args=["-O3", "-std=c++2a"],
)

ext_modules = [pyimmer]


def build():
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
