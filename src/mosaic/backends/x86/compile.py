import pathlib
import subprocess
from typing import Collection
from pathlib import Path

from loguru import logger

FLAGS = [
    "-std=c++17",
    "-O3",
    "-g",
    "-march=native",
    "-fno-omit-frame-pointer",
    "-fno-exceptions",
    "-mavx2",
    "-msse4",
    "-mfma",
    "-maes",
    # "-shared",
    "-fPIC",
    "-Wall",
    "-Wno-deprecated",
    "-Wno-unused-function",
    "-Wno-multichar",
    "-Wno-format",
]


def compile_source_file_to_assembly(source_file: Path, include_paths: Collection[Path], output_object_file: Path):
    output_object_file.unlink(missing_ok=True)
    output_object_file = str(output_object_file)

    source_file = str(source_file)
    include_paths = [str(include_path) for include_path in include_paths]

    command = ["g++", "-c", source_file, *include_paths, *FLAGS, "-S", "-fverbose-asm", "-o", output_object_file]
    logger.info(f"Compile Source Code to Assembly: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0


def compile_source_file_to_object_file(source_file: Path, include_paths: Collection[Path], output_object_file: Path):
    output_object_file.unlink(missing_ok=True)
    output_object_file = str(output_object_file)

    source_file = str(source_file)
    include_paths = [str(include_path) for include_path in include_paths]

    command = ["g++", "-c", source_file, *include_paths, *FLAGS, "-o", output_object_file]
    logger.info(f"Compile Source Code to Object File: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0


def compile_shared_library(kernel_source_file: pathlib.Path, enable_tracy=False):
    output_path = kernel_source_file.parent
    include_paths = [
        "-I",
        str(output_path),
    ]
    kernel_object_file = kernel_source_file.with_suffix(".o")
    object_files = [kernel_object_file]

    if enable_tracy:
        FLAGS.append("-DTRACY_ENABLE")
        tracy_object_file = output_path / "tracy.o"

        tracy_include_path = ["-I", "vendor/tracy/public"]
        include_paths.extend(tracy_include_path)

        compile_source_file_to_object_file(
            Path("vendor/tracy/public/TracyClient.cpp"), tracy_include_path, tracy_object_file
        )
        object_files.append(tracy_object_file)

    compile_source_file_to_object_file(kernel_source_file, include_paths, kernel_object_file)

    object_files = [str(object_file) for object_file in object_files]
    shared_library = kernel_source_file.with_suffix(".so")
    shared_library.unlink(missing_ok=True)
    shared_library = str(shared_library)
    command = ["g++", "-fPIC", "-shared", "-rdynamic", *object_files, "-o", shared_library]
    logger.info(f"Compile Object Files to Shared Library: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    kernel_assembly_file = kernel_source_file.with_suffix(".s")
    compile_source_file_to_assembly(kernel_source_file, include_paths, kernel_assembly_file)

    return shared_library
