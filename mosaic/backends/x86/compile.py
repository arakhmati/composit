import subprocess

from loguru import logger

FLAGS = [
    "-std=c2x",
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


def compile_shared_library(test_output_path, kernel_name):
    source_file = str(test_output_path / f"{kernel_name}.c")
    assembly_file = test_output_path / f"{kernel_name}.s"
    assembly_file.unlink(missing_ok=True)
    assembly_file = str(assembly_file)
    command = [
        "gcc",
        source_file,
        "-I",
        str(test_output_path),
        *FLAGS,
        "-S",
        "-fverbose-asm",
        "-o",
        assembly_file,
    ]
    logger.info(f"Compile Source Code to Assembly: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    shared_library_file = test_output_path / f"{kernel_name}.so"
    shared_library_file.unlink(missing_ok=True)
    shared_library_file = str(shared_library_file)
    command = ["gcc", assembly_file, "-fPIC", "-shared", "-o", shared_library_file]
    logger.info(f"Compile Assembly to Binary: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    return shared_library_file
