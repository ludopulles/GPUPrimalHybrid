from sage.all import floor, round # type: ignore #noqa
import numpy as np # type: ignore #noqa
import matplotlib.pyplot as plt # type: ignore #noqa
from typing import Tuple
from sage.all import parent, ZZ, QQ # type: ignore #noqa


def approx_nu(nu: float) -> Tuple[int, int]:
    # approximate a real nu to a rational, within a 1% error
    app = QQ(round(nu * 100)/100)
    x = int(app.numerator())
    y = int(app.denominator())
    return x, y


def balance(e, q=None):
    """
    Return a representation of `e` with elements balanced between `-q/2` and `q/2`

    :param e: a vector, polynomial or scalar
    :param q: optional modulus, if not present this function tries to recover it from `e`

    :returns: a vector, polynomial or scalar over/in the integers
    """
    try:
        p = parent(e).change_ring(ZZ)
        return p([balance(e_, q=q) for e_ in e])
    except (TypeError, AttributeError):
        if q is None:
            try:
                q = parent(e).order()
            except AttributeError:
                q = parent(e).base_ring().order()
        e = ZZ(e)
        e = e % q
        return ZZ(e-q) if e > q//2 else ZZ(e)


def _round(x: float):
    return floor(x + .5)


def round_down(x: int, q: int, p: float):
    """ Round from mod q to mod p, where q >= p """
    return _round(balance(x, q=q) * p / q)


def histogram(data, bins, fn, curve=None, scatter=None):
    fig, axs = plt.subplots(1, 1)

    import collections
    counter = collections.Counter(list(data))
    plt.plot(
        sorted(counter.keys()),
        [counter[x] for x in sorted(counter.keys())]
    )

    min_data = min(data)
    max_data = max(data)
    if curve:
        x = np.linspace(min_data, max_data)
        plt.plot(x, curve(x), lw=2)
    if scatter:
        plt.plot(scatter[0], scatter[1], lw=2)
    plt.savefig(fn)
