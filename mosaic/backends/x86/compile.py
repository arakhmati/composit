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


def compile_source_file_to_object_file(source_file: Path, include_paths: Collection[Path], output_object_file: Path):
    output_object_file.unlink(missing_ok=True)
    output_object_file = str(output_object_file)

    source_file = str(source_file)
    include_paths = [str(include_path) for include_path in include_paths]

    command = ["g++", "-c", source_file, *include_paths, *FLAGS, "-o", output_object_file]
    logger.info(f"Compile source Code to object file: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0


def compile_shared_library(test_output_path, kernel_name, enable_tracy=False):
    include_paths = [
        "-I",
        str(test_output_path),
    ]
    kernel_source_file = test_output_path / f"{kernel_name}.cpp"
    kernel_object_file = test_output_path / f"{kernel_name}.o"
    object_files = [kernel_object_file]

    if enable_tracy:
        FLAGS.append("-DTRACY_ENABLE")
        tracy_object_file = test_output_path / "tracy.o"

        tracy_include_path = ["-I", "vendor/tracy/public"]
        include_paths.extend(tracy_include_path)

        compile_source_file_to_object_file(
            Path("vendor/tracy/public/TracyClient.cpp"), tracy_include_path, tracy_object_file
        )
        object_files.append(tracy_object_file)

    compile_source_file_to_object_file(kernel_source_file, include_paths, kernel_object_file)

    object_files = [str(object_file) for object_file in object_files]
    shared_library = test_output_path / f"{kernel_name}.so"
    shared_library.unlink(missing_ok=True)
    shared_library = str(shared_library)
    command = ["g++", "-fPIC", "-shared", "-rdynamic", *object_files, "-o", shared_library]
    logger.info(f"Compile Assembly to Binary: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    return shared_library
