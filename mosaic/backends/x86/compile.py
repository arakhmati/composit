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
    base_command = ["gcc", source_file, "-I", str(test_output_path), *FLAGS]

    assembly_file = test_output_path / f"{kernel_name}.s"
    assembly_file.unlink(missing_ok=True)
    assembly_file = str(assembly_file)
    assembly_command = base_command + [
        "-S",
        "-fverbose-asm",
        "-o",
        assembly_file,
    ]
    logger.info(f"Compile Source Code to Assembly: \"{' '.join(assembly_command)}\"")
    result = subprocess.run(assembly_command)
    assert result.returncode == 0

    shared_library_file = test_output_path / f"{kernel_name}.so"
    shared_library_file.unlink(missing_ok=True)
    shared_library_file = str(shared_library_file)
    shared_library_command = base_command + [
        "-shared",
        "-o",
        shared_library_file,
    ]
    logger.info(f"Compile Source Code to Shared Library: \"{' '.join(shared_library_command)}\"")
    result = subprocess.run(shared_library_command)
    assert result.returncode == 0

    return shared_library_file
