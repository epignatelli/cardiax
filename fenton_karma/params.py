from typing import NamedTuple


_maxfloat = 1e6

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


paramset_3 = Params(3.33,
                    19.6,
                    1250,
                    870,
                    41,
                    0.25,
                    12.5,
                    33.33,
                    29,
                    10,
                    0.85,
                    0.13,
                    0.04,
                    1)

def _params1():
    params = {
        "tau_v_plus": 3.33,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 19.6,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 1000,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": 667,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": 11,  # Slow_inward_current_w_gate (ms)
        "tau_d": NotImplemented,  # excitability
        "tau_0": 8.3,  # Slow_outward_current (ms)
        "tau_r": 50,  # Slow_outward_current (ms)
        "tau_si": 45,  # Slow_inward_current (ms)
        "k": 10,  # Slow_inward_current (dimensionless)
        "V_csi": 0.85,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": 0.0055,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }
    return params


def params1a():
    params = _params1()
    params["tau_d"] = 0.41
    return params


def params1b():
    params = _params1()
    params["tau_d"] = 0.392
    return params


def params1c():
    params = _params1()
    params["tau_d"] = 0.381
    return params


def params1d():
    params = _params1()
    params["tau_d"] = 0.36
    return params


def params1e():
    params = _params1()
    params["tau_d"] = 0.25
    return params


def params2():
    return {
        "tau_v_plus": 10,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 10,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 10,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": _maxfloat,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": _maxfloat,  # Slow_inward_current_w_gate (ms)
        "tau_d": 0.25,  # excitability
        "tau_0": 10,  # Slow_outward_current (ms)
        "tau_r": 190,  # Slow_outward_current (ms)
        "tau_si": _maxfloat,  # Slow_inward_current (ms)
        "k": 100000,  # Slow_inward_current (dimensionless)
        "V_csi": _maxfloat,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": _maxfloat,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def params3():
    return {
        "tau_v_plus": 3.33,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 19.6,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 1250,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": 870,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": 41,  # Slow_inward_current_w_gate (ms)
        "tau_d": 0.25,  # excitability
        "tau_0": 12.5,  # Slow_outward_current (ms)
        "tau_r": 33.33,  # Slow_outward_current (ms)
        "tau_si": 29,  # Slow_inward_current (ms)
        "k": 10,  # Slow_inward_current (dimensionless)
        "V_csi": 0.85,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": 0.04,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def _params4():
    return {
        "tau_v_plus": 3.33,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 15.6,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 5,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": 350,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": 80,  # Slow_inward_current_w_gate (ms)
        "tau_d": NotImplemented,  # excitability
        "tau_0": 9,  # Slow_outward_current (ms)
        "tau_r": 34,  # Slow_outward_current (ms)
        "tau_si": 26.5,  # Slow_inward_current (ms)
        "k": 15,  # Slow_inward_current (dimensionless)
        "V_csi": 0.45,  # Slow_inward_current (dimensionless)
        "V_c": 0.15,
        "V_v": 0.04,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def params4a():
    params = _params4()
    params["tau_d"] = 0.407
    return params


def params4b():
    params = _params4()
    params["tau_d"] = 0.405
    return params


def params4c():
    params = _params4()
    params["tau_d"] = 0.4
    return params


def params5():
    return {
        "tau_v_plus": 3.33,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 12,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 2,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": 1000,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": 100,  # Slow_inward_current_w_gate (ms)
        "tau_d": 0.362,  # excitability
        "tau_0": 5,  # Slow_outward_current (ms)
        "tau_r": 33.33,  # Slow_outward_current (ms)
        "tau_si": 29,  # Slow_inward_current (ms)
        "k": 15,  # Slow_inward_current (dimensionless)
        "V_csi": 0.7,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": 0.04,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def params6():
    return {
        "tau_v_plus": 3.33,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 9,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 8,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": 250,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": 60,  # Slow_inward_current_w_gate (ms)
        "tau_d": 0.395,  # excitability
        "tau_0": 9,  # Slow_outward_current (ms)
        "tau_r": 33.33,  # Slow_outward_current (ms)
        "tau_si": 29,  # Slow_inward_current (ms)
        "k": 15,  # Slow_inward_current (dimensionless)
        "V_csi": 0.5,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": 0.04,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def params7():
    return {
        "tau_v_plus": 10,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 7,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 7,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": _maxfloat,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": _maxfloat,  # Slow_inward_current_w_gate (ms)
        "tau_d": 0.25,  # excitability
        "tau_0": 12,  # Slow_outward_current (ms)
        "tau_r": 100,  # Slow_outward_current (ms)
        "tau_si": _maxfloat,  # Slow_inward_current (ms)
        "k": _maxfloat,  # Slow_inward_current (dimensionless)
        "V_csi": _maxfloat,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": _maxfloat,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def params8():
    return {
        "tau_v_plus": 13.03,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 19.6,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 1250,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": 800,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": 40,  # Slow_inward_current_w_gate (ms)
        "tau_d": 0.45,  # excitability
        "tau_0": 12.5,  # Slow_outward_current (ms)
        "tau_r": 33.25,  # Slow_outward_current (ms)
        "tau_si": 29,  # Slow_inward_current (ms)
        "k": 10,  # Slow_inward_current (dimensionless)
        "V_csi": 0.85,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": 0.04,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def params9():
    return {
        "tau_v_plus": 3.33,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 15,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 2,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": 670,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": 61,  # Slow_inward_current_w_gate (ms)
        "tau_d": 0.25,  # excitability
        "tau_0": 12.5,  # Slow_outward_current (ms)
        "tau_r": 28,  # Slow_outward_current (ms)
        "tau_si": 29,  # Slow_inward_current (ms)
        "k": 10,  # Slow_inward_current (dimensionless)
        "V_csi": 0.45,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": 0.05,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def params10():
    return {
        "tau_v_plus": 10,  # Fast_inward_current_v_gate (ms)
        "tau_v1_minus": 40,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 333,  # Fast_inward_current_v_gate (ms)
        "tau_w_plus": 1000,  # Slow_inward_current_w_gate (ms)
        "tau_w_minus": 65,  # Slow_inward_current_w_gate (ms)
        "tau_d": 0.115,  # excitability
        "tau_0": 12.5,  # Slow_outward_current (ms)
        "tau_r": 25,  # Slow_outward_current (ms)
        "tau_si": 22.22,  # Slow_inward_current (ms)
        "k": 10,  # Slow_inward_current (dimensionless)
        "V_csi": 0.85,  # Slow_inward_current (dimensionless)
        "V_c": 0.13,
        "V_v": 0.0025,
        "Cm": 1,  # membrane capacitance (microF / cm^2)
    }


def params_test():
    return {
        "Cm": 1,  # membrane capacitance (microF / cm^2) measured in micro Farads
        "V_c": 0.16,
        "V_v": 0.16,
        "tau_d": 0.125,  # excitability (ms)
        "tau_v1_minus": 82.5,  # Fast_inward_current_v_gate (ms)
        "tau_v2_minus": 60,  # Fast_inward_current_v_gate (ms)
        "tau_v_plus": 5.75,  # Fast_inward_current_v_gate (ms)
        "tau_0": 32.5,  # Slow_outward_current (ms)
        "tau_r": 70,  # Slow_outward_current (ms)
        "tau_si": 114,  # Slow_inward_current (ms)
        "V_csi": 0.85,  # Slow_inward_current (dimensionless)
        "k": 10,  # Slow_inward_current (dimensionless)
        "tau_w_minus": 400,  # Slow_inward_current_w_gate (ms)
        "tau_w_plus": 300,  # Slow_inward_current_w_gate (ms)
        "D": 0.05,  # diffusivity (cm^2/ms)
    }
