import pickle
import zlib


def deterministic_hash(obj):
    bytes = pickle.dumps(obj)
    return zlib.adler32(bytes)
