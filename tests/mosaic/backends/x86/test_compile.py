import ctypes
import os
import subprocess

import pytest

from mosaic.backends.x86.compile import compile_shared_library


def test_compile_and_load_shared_library(tmp_path):
    output = tmp_path / "directory with spaces"
    output.mkdir()
    source = output / "kernel.cpp"
    source.write_text('extern "C" int add(int a, int b) { return a + b; }\n')
    flags = ["-DTEST_BUILD"]
    library_path = compile_shared_library(source, flags=flags, compile_assembly=True)
    assert flags == ["-DTEST_BUILD"]
    assert library_path.endswith(".dll" if os.name == "nt" else ".so")
    assert source.with_suffix(".s").exists()
    library = ctypes.CDLL(library_path)
    library.add.argtypes = [ctypes.c_int, ctypes.c_int]
    library.add.restype = ctypes.c_int
    assert library.add(19, 23) == 42


def test_compiler_failure_is_reported(tmp_path):
    source = tmp_path / "broken.cpp"
    source.write_text("this is not valid C++\n")
    with pytest.raises(subprocess.CalledProcessError):
        compile_shared_library(source)
