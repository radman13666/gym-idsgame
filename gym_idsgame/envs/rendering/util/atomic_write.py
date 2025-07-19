import os
import tempfile
from contextlib import contextmanager

@contextmanager
def atomic_write(filepath, mode='w', encoding='utf-8'):
    dirpath = os.path.dirname(filepath)
    with tempfile.NamedTemporaryFile(mode=mode, encoding=encoding, dir=dirpath, delete=False) as tmp:
        tempname = tmp.name
        try:
            yield tmp
            tmp.flush()
            os.fsync(tmp.fileno())
        finally:
            tmp.close()
            os.replace(tempname, filepath)
