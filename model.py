from jax import random, pmap
import jax.numpy as np

class FentonKarma(object):
    def __init__(self):
        # Geometry
        self.height = 128  # units
        self.width = 128  # units
        self.dx = dy = 0.025  # mm 
        self.d2 = dx ** 2

        # state
        self.voltage = np.zeros((self.height, self.width))
        self.fast_current = np.ones(self.height, self.width)
        self.slow_current = np.ones(self.height, self.width)
        self.time = 0

        # parameters
        self.cm = 1  # capacitance
        self.v_c = 0.13
        self.v_v = 0.04
        self.tau_d = 0.395
        self.tau_v1_minus = 9
        self.tau_v2_minus = 8
        self.tau_v_plus = 3.33
        self.tau_0 = 9
        self.tau_r = 33.33
        self.tau_si = 29
        self.v_csi = 0.5
        self.k = 15
        self.tau_w_minus = 60
        self.tau_w_plus = 250
        self.Dx = 0.001
        self.Dy = 0.001

        # integration
        self.dt = 0.1

        # stimulus
        self.stimulus = np.zeros((height, width))
        self.stimulus[:40, :40] = -0.2

    def step(self):



    def apply_stimulus(self):
        if 

    def gradient(self, field):
        



class Stimulus(object):
    def __init__(self, value=-0.2, shape=(128, 128)):
        self.modulus = value
        self.shape = shape
        self.protocol = Procotol()

class Protocol(object):
    def __init__(self, protocol="single"):
        self.protocol = protocol
        self.start = 0
        self.end = -1
        self.step = -1


