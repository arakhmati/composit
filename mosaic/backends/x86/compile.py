import subprocess

from loguru import logger

FLAGS = [
    "-std=c17",
    "-O3",
    "-march=native",
    "-fno-exceptions",
    "-mavx2",
    "-msse4",
    "-mfma",
    "-maes",
    "-shared",
    "-fPIC",
    "-Wall",
    "-Wno-deprecated",
    "-Wno-unused-function",
    "-Wno-multichar",
    "-Wno-format",
]


def compile_shared_library(test_output_path, module):
    kernel_name = module.__name__.split(".")[-1]
    source_file = str(test_output_path / f"{kernel_name}.c")
    assembly = test_output_path / f"{kernel_name}.s"
    assembly.unlink(missing_ok=True)
    assembly = str(assembly)
    command = ["gcc", source_file, "-I", str(test_output_path), *FLAGS, "-S", "-fverbose-asm", "-o", assembly]
    logger.info(f"Compile Source Code to Assembly: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    shared_library = test_output_path / f"{kernel_name}.so"
    shared_library.unlink(missing_ok=True)
    shared_library = str(shared_library)
    command = ["gcc", assembly, "-fPIC", "-shared", "-o", shared_library]
    logger.info(f"Compile Assembly to Binary: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    return shared_library
