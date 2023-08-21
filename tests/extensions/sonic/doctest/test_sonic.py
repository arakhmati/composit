import pytest

import pathlib
import subprocess

TEST_DIR = pathlib.Path(__file__).parent
INCLUDE_DIR = pathlib.Path(".") / "src" / "extensions" / "sonic" / "include"

INCLUDES = ["-I", INCLUDE_DIR]
FLAGS = [
    "-std=c++2a",
    "-fconcepts",
    "-fPIC",
    "-march=native",
    "-mavx2",
    "-msse4",
    "-mfma",
    "-maes",
    "-fno-omit-frame-pointer",
    "-fno-exceptions",
    "-Wall",
    "-Wno-deprecated",
    "-Wno-unused-function",
    "-Wno-multichar",
    "-Wno-format",
    "-lpthread",
]

CPP_FILES = TEST_DIR.glob("test_*.cpp")


@pytest.mark.parametrize("cpp_file_name", CPP_FILES)
def test(tmp_path, cpp_file_name):
    executable = tmp_path / cpp_file_name.stem
    compilation_command = ["g++", cpp_file_name, *INCLUDES, *FLAGS, "-o", executable]

    status = subprocess.run(compilation_command)
    assert status.returncode == 0

    status = subprocess.run(executable)
    assert status.returncode == 0
