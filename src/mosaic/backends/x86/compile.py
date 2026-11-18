import os
import pathlib
import subprocess
from typing import Collection
from pathlib import Path

from loguru import logger

FLAGS = [
    "-std=c++2a",
    "-O3",
    "-march=native",
    "-fno-exceptions",
    "-mavx2",
    "-msse4",
    "-mfma",
    "-maes",
    "-Wall",
    "-Wpedantic",
    "-Wextra",
    "-Wno-deprecated",
    "-Wno-unused-function",
    "-Wno-multichar",
    "-Wno-format",
]


if os.name != "nt":
    FLAGS.append("-fPIC")


def process_include_paths(include_paths):
    def generator():
        for include_path in include_paths:
            yield "-I"
            yield include_path

    return list(generator())


def compile_source_file_to_assembly(source_file: Path, include_paths: Collection[str], flags, output_object_file: Path):
    output_object_file.unlink(missing_ok=True)
    output_object_file = str(output_object_file)

    source_file = str(source_file)

    command = [
        os.environ.get("CXX", "g++"),
        "-c",
        source_file,
        *include_paths,
        *flags,
        "-S",
        "-fverbose-asm",
        "-o",
        output_object_file,
    ]
    logger.info(f"Compile Source Code to Assembly: \"{' '.join(command)}\"")
    subprocess.run(command, check=True)


def compile_source_file_to_object_file(
    source_file: Path, include_paths: Collection[str], flags, output_object_file: Path
):
    output_object_file.unlink(missing_ok=True)
    output_object_file = str(output_object_file)

    source_file = str(source_file)

    command = [os.environ.get("CXX", "g++"), "-c", source_file, *include_paths, *flags, "-o", output_object_file]
    logger.info(f"Compile Source Code to Object File: \"{' '.join(command)}\"")
    subprocess.run(command, check=True)


def compile_shared_library(
    source_file: pathlib.Path,
    *,
    include_paths=None,
    flags=None,
    enable_tracy=False,
    compile_assembly=False,
):
    source_file = source_file.resolve()
    output_path = source_file.parent

    object_file = source_file.with_suffix(".o")
    object_file.unlink(missing_ok=True)

    object_files = [object_file]

    if include_paths is None:
        include_paths = []
    include_paths = [str(output_path)] + [str(include_path) for include_path in include_paths]
    for include_path in include_paths:
        assert pathlib.Path(include_path).exists(), f"{include_path} doesn't exist"

    flags = list(flags or []) + FLAGS

    if enable_tracy:
        flags += ["-DTRACY_ENABLE"]
        tracy_object_file = output_path / "tracy.o"

        tracy_include_path = ["-I", "vendor/tracy/public"]
        include_paths.append("vendor/tracy/public")

        compile_source_file_to_object_file(
            Path("vendor/tracy/public/TracyClient.cpp"), tracy_include_path, flags, tracy_object_file
        )
        object_files.append(tracy_object_file)

    compile_source_file_to_object_file(source_file, process_include_paths(include_paths), flags, object_file)

    object_files = [str(object_file) for object_file in object_files]
    shared_library = source_file.with_suffix(".dll" if os.name == "nt" else ".so")
    shared_library.unlink(missing_ok=True)
    shared_library = str(shared_library)
    # MinGW exports the C entry points and bundles its runtime into the DLL.
    link_flags = ["-static", "-Wl,--export-all-symbols"] if os.name == "nt" else ["-fPIC", "-rdynamic"]
    command = [os.environ.get("CXX", "g++"), "-shared", *link_flags, *object_files, "-o", shared_library]
    logger.info(f"Compile Object Files to Shared Library: \"{' '.join(command)}\"")
    subprocess.run(command, check=True)

    if compile_assembly:
        assembly_file = source_file.with_suffix(".s")
        compile_source_file_to_assembly(source_file, process_include_paths(include_paths), flags, assembly_file)

    return shared_library
