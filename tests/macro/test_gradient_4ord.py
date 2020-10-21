import jax.numpy as np


def intervals():
    pos = [np.nan, np.nan]
    a = np.arange(10)
    m2 = np.hstack([pos, a[:-4], pos])
    m1 = np.hstack([pos, a[1:-3], pos])
    p1 = np.hstack([pos, a[3:-1], pos])
    p2 = np.hstack([pos, a[4:], pos])

    print(np.vstack([m2, m1, p1, p2]))
    return
