from typing import NamedTuple


class Params(NamedTuple):
    tau_v_plus: float
    tau_v1_minus: float
    tau_v2_minus: int
    tau_w_plus: int
    tau_w_minus: int
    tau_d: float
    tau_0: float
    tau_r: int
    tau_si: int
    k: int
    V_csi: float
    V_c: float
    V_v: float
    Cm: float


MAXFLOAT = 1e6

PARAMSET_1A = Params(
    3.33, 19.6, 1000, 667, 11, 0.41, 8.3, 50, 45, 10, 0.85, 0.13, 0.0055, 1
)

PARAMSET_1B = Params(
    3.33, 19.6, 1000, 667, 11, 0.392, 8.3, 50, 45, 10, 0.85, 0.13, 0.0055, 1
)

PARAMSET_1C = Params(
    3.33, 19.6, 1000, 667, 11, 0.381, 8.3, 50, 45, 10, 0.85, 0.13, 0.0055, 1
)

PARAMSET_1D = Params(
    3.33, 19.6, 1000, 667, 11, 0.36, 8.3, 50, 45, 10, 0.85, 0.13, 0.0055, 1
)

PARAMSET_1E = Params(
    3.33, 19.6, 1000, 667, 11, 0.25, 8.3, 50, 45, 10, 0.85, 0.13, 0.0055, 1
)

PARAMSET_2 = Params(
    10,
    10,
    10,
    MAXFLOAT,
    MAXFLOAT,
    0.25,
    10,
    190,
    MAXFLOAT,
    100000,
    MAXFLOAT,
    0.13,
    MAXFLOAT,
    1,
)

PARAMSET_3 = Params(
    3.33, 19.6, 1250, 870, 41, 0.25, 12.5, 33.33, 29, 10, 0.85, 0.13, 0.04, 1
)

PARAMSET_4A = Params(
    3.33, 15.6, 5, 350, 80, 0.407, 9, 34, 26.5, 15, 0.45, 0.15, 0.04, 1
)

PARAMSET_4B = Params(
    3.33, 15.6, 5, 350, 80, 0.405, 9, 34, 26.5, 15, 0.45, 0.15, 0.04, 1
)

PARAMSET_4C = Params(3.33, 15.6, 5, 350, 80, 0.4, 9, 34, 26.5, 15, 0.45, 0.15, 0.04, 1)

PARAMSET_5 = Params(3.33, 12, 2, 1000, 100, 0.362, 5, 33.33, 29, 15, 0.7, 0.13, 0.04, 1)

PARAMSET_6 = Params(3.33, 9, 8, 250, 60, 0.395, 9, 33.33, 29, 15, 0.5, 0.13, 0.04, 1)

PARAMSET_7 = Params(
    10,
    7,
    7,
    MAXFLOAT,
    MAXFLOAT,
    0.25,
    12,
    100,
    MAXFLOAT,
    MAXFLOAT,
    MAXFLOAT,
    0.13,
    MAXFLOAT,
    1,
)

PARAMSET_8 = Params(
    13.03, 19.06, 1250, 800, 40, 0.45, 12.5, 33.25, 29, 10, 0.85, 0.13, 0.04, 1
)

PARAMSET_9 = Params(3.33, 15, 2, 670, 61, 0.25, 12.5, 28, 29, 10, 0.45, 0.13, 0.05, 1)

PARAMSET_10 = Params(
    10, 40, 333, 1000, 65, 0.115, 12.5, 25, 22.22, 10, 0.85, 0.13, 0.0025, 1
)
