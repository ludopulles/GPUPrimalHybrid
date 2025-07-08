try:
    from config import NPROC # type: ignore #noqa
except ImportError:
    import multiprocessing
    NPROC = int(multiprocessing.cpu_count()) - 2

try:
    from config import FLATTER_MODE # type: ignore #noqa
except ImportError:
    FLATTER_MODE = 3

try:
    from config import FLATTER_HOST_DIR # type: ignore #noqa
except ImportError:
    FLATTER_HOST_DIR = "/usr/local/bin"

try:
    from config import CACHEDIR # type: ignore #noqa
except ImportError:
    CACHEDIR = "cachedir"

try:
    from config import MAX_BLOCK_SIZE # type: ignore #noqa
except ImportError:
    MAX_BLOCK_SIZE = 60

try:
    from config import MAX_REPEATS # type: ignore #noqa
except ImportError:
    MAX_REPEATS = 1000**3
