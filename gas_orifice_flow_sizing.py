# Size orifice for a 75 psi pressure drop, 0.25 in line, helium as the gas, and a 0.336 ft ** 3 / s
import fluids as fl
import scipy.optimize as sp
from pint import UnitRegistry
import numpy as np
u = UnitRegistry()
HE_DynamicViscosity_DEFAULT = (2.10E-5 * u.Pa * u.s).to('lb / (ft * s)')
N2_DynamicViscosity_DEFAULT = (19.84E-6 * u.Pa * u.s)
def run():
    D = 0.25 * u.inch

    # rho = 2.11 * u.lb / u.ft ** 3 # density of helium at 3000 psi
    rho = 7.3935 * u.lb / u.ft ** 3
    Q = 0.336 * u.ft ** 3 / u.s


    Do = (fl.differential_pressure_meter_solver(D=D.to('m').magnitude,
                                          rho=rho.to('kg / m ** 3').magnitude,
                                          mu=N2_DynamicViscosity_DEFAULT.magnitude,
                                          k=1.4,
                                          P1=(3000 * u.psi).to('Pa').magnitude,
                                          P2=(2900 * u.psi).to('Pa').magnitude,
                                          m=(Q * rho).to('kg / s').magnitude,
                                          meter_type='ISO 5167 orifice',
                                          taps='flange')*u.m).to('in')
    # DP = 80 * u.psi
    # Y = 0.99
    # guessarr = (0.5, 1.0, 0.2)
    # data = (D, DP, rho, Q, Y)
    # sol = sp.fsolve(equations, guessarr, args=data)
    # d = (1 / ((1 / D ** 4) + DP / rho * (0.065607 * C * Y / Q) ** 2)) ** 0.25
    # print(str(sol))
    print(str(Do))
    print('Done')

# Didn't work
# def equations(p, *data):
#     D, DP, rho, Q, Y = data
#     C, Kf, d = p
#     g = 32.174 * u.ft / u.s ** 2
#     return (
#         C - fl.C_Reader_Harris_Gallagher(D=D.to('m').magnitude, Do=(d * u.inch).to('m').magnitude,
#                                          rho=rho.to('kg / m ** 3').magnitude,
#                                          mu=HE_DynamicViscosity_DEFAULT.to('Pa * s').magnitude,
#                                          m=(Q * rho).to('kg / s').magnitude,
#                                          taps='flange'),
#         Kf - (C / np.sqrt(np.abs((1 - (d / D.magnitude) ** 4)))),
#         Q.magnitude - (Kf * np.pi / 4 * (d * u.inch).to('ft') ** 2 * Y / 12 * np.sqrt(2 * g * DP.to('lbf / ft ** 2') / rho )).magnitude
#      )

if __name__ == '__main__':
    run()