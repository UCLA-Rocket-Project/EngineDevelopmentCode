import scipy.optimize as sp
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import fluids as fl
import thermo
import pint
ureg = pint.UnitRegistry()
from RP1_PROPERTIES import RP1_PROPERTIES

# Source: https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html
RP1_DynamicViscosity_DEFAULT = 11E-4 * ureg.lb / (ureg.ft * ureg.s)
LOX_DynamicViscosity_DEFAULT = (284.9E-6 * ureg.Pa * ureg.s).to('lb / (ft * s)')
# Source: https://www.sciencedirect.com/science/article/pii/S0011227507001506

RP1_Density_DEFAULT = 51.4 * ureg.lb / (ureg.ft ** 3)
LOX_Density_DEFAULT = 75.038409 * ureg.lb / (ureg.ft ** 3)
# Source: https://www.sciencedirect.com/science/article/pii/S0011227507001506

ABS_ROUGHNESS_ALUMINUM = 6.56E-6 * ureg.inch# Source: https://www.engineeringtoolbox.com/surface-roughness-ventilation-ducts-d_209.html
ABS_ROUGHNESS_SS = 0.0197E-3 * ureg.inch
PERCENT_FFC = 11.5 # percent fuel film coolant
G_const = 32.174

def run():
    # w2 components
    # CdA_PRO_PFTI
    #   Lines leading to tank inlet, check valve, helium
    #   DP < 5, ignore
    # CdA_PFTI_PFT
    #   Fuel tank diffuser section to  ullage start, helium
    #   DP < 5, ignore
    # CdA_PFT_PFTO
    #   Fuel tank gas/liquid line, converging section to line diameter
    # CdA_PFTO_PMFVI
    #   Lines leading up to main fuel valve inlet
    # CdA_PMFVI_PFMVO
    #   CdA of main fuel valve
    # CdA_PMFVO_PFCI
    #   Lines leading to film coolant inlet branch
    w2guess = 0.99 * (ureg.lb / ureg.s)
    D2 = 0.5 * ureg.inch
    Re2 = (4 / np.pi) * (w2guess / (D2.to('ft') * RP1_DynamicViscosity_DEFAULT))
    f2 = fl.friction_factor(Re2, eD=ABS_ROUGHNESS_SS/D2)

    Tank_Diameter = 1 * ureg.ft
    K_PFT_PFTO = fl.fittings.contraction_conical(Tank_Diameter.to('m'), D2, f2, l=(4*ureg.inch).to('m'))
    CdA_PFT_PFTO = K_to_CdA(K_PFT_PFTO,  D=D2)

    K_PFTO_PMFVI = fl.K_from_f(f2, L=(5*ureg.ft).to('in'), D=D2) # line losses
    K_PFTO_PMFVI += fl.fittings.bend_rounded(D2, angle=90, fd=f2, bend_diameters=5)
    K_PFTO_PMFVI += fl.fittings.bend_rounded(D2, angle=90, fd=f2, bend_diameters=5)
    K_PFTO_PMFVI += fl.fittings.bend_rounded(D2, angle=90, fd=f2, bend_diameters=5)
    CdA_PFTO_PMFVI = K_to_CdA(K_PFTO_PMFVI, D=D2)

    K_PMFVI_PMFVO = fl.fittings.K_ball_valve_Crane(D1=D2,D2=D2, angle=0, fd=f2)
    CdA_PMFVI_PMFVO = K_to_CdA(K_PMFVI_PMFVO, D=D2)

    K_PMFVO_PFCI = fl.K_from_f(f2, L=(3.5*ureg.ft).to('in'), D=D2)
    K_PMFVO_PFCI += fl.fittings.bend_rounded(D2, angle=90, fd=f2, bend_diameters=5)
    K_PMFVO_PFCI += fl.fittings.bend_rounded(D2, angle=90, fd=f2, bend_diameters=5)

    # Nozzle outlet helicoil, 2 spirals
    CHAMBER_THICKNESS = 0.070 * ureg.inch
    DIA_NOZZLE_EXIT = 3.380 * ureg.inch + 2 *CHAMBER_THICKNESS
    COOLING_CHANNEL_HEIGHT = 0.08 * ureg.inch
    DH_COOLING = 2 * COOLING_CHANNEL_HEIGHT # Hydraulic Diameter, approx
    PITCH_NOZZLE_OUTLET = 1.8 * ureg.inch   # Spacing between each coil
    K_PMFVO_PFCI += fl.change_K_basis(
        fl.fittings.spiral(DH_COOLING, DIA_NOZZLE_EXIT/2, DIA_NOZZLE_EXIT/2 * 0.66, PITCH_NOZZLE_OUTLET, f2),
        DH_COOLING, D2)
    DIA_THROAT = 1.658 * ureg.inch + 2 * CHAMBER_THICKNESS
    PITCH_THROAT = 0.9 * ureg.inch
    K_PMFVO_PFCI += fl.change_K_basis(
        fl.fittings.spiral(DH_COOLING, DIA_NOZZLE_EXIT/2 * 0.66, DIA_THROAT / 2, PITCH_THROAT, f2),
        DH_COOLING, D2)

    DIA_CHAMBER = 4.06 * ureg.inch
    K_PMFVO_PFCI += fl.change_K_basis(
        fl.fittings.spiral(DH_COOLING, DIA_THROAT / 2, DIA_CHAMBER / 2, PITCH_THROAT, f2),
        DH_COOLING, D2)


    CdA_PMFVO_PFCI = K_to_CdA(K_PMFVO_PFCI, D=D2)

    # w3 components
    # CdA_PFCI_PFM
    #   Lines leading to fuel manifold
    # CdA_PFM_PC
    #   CdA of fuel injector
    w3guess = w2guess * ((100-PERCENT_FFC)/100.0)
    D3 = D2
    Re3 = (4 / np.pi) * (w3guess / (D3.to('ft') * RP1_DynamicViscosity_DEFAULT))
    f3 = fl.friction_factor(Re3, eD=ABS_ROUGHNESS_SS/D3)

    K_PFCI_PFM = fl.K_from_f(f3, L=(3.5*ureg.ft).to('in'), D=D2)
    K_PFCI_PFM += fl.fittings.bend_rounded(D3, angle=90, fd=f3, bend_diameters=5)
    CdA_PFCI_PFM = K_to_CdA(K_PFCI_PFM, D=D3)

    # Ares 2017-2018 INJECTOR GEOMETRIES, Dave Crisalli design
    # NUM_FUEL_HOLES = 16
    # DIA_FUEL_HOLES = 0.052 * ureg.inch
    # NUM_FUEL_SHOWERHEAD_HOLES = 8
    # DIA_FUEL_SHOWEHEAD_HOLES = 0.029 * ureg.inch
    # NUM_HEAD_FFC_HOLES = 16
    # DIA_HEAD_FFC_HOLES = 0.029 * ureg.inch
    # Cd_PFM_PC = 0.78 # Determined empirically through waterflow data
    # A_PFM_PC = np.pi / 4 * (NUM_FUEL_HOLES * DIA_FUEL_HOLES ** 2 + NUM_HEAD_FFC_HOLES * DIA_HEAD_FFC_HOLES ** 2 +
    #                         NUM_FUEL_SHOWERHEAD_HOLES * DIA_FUEL_SHOWEHEAD_HOLES ** 2)
    # CdA_PFM_PC = Cd_PFM_PC * A_PFM_PC
    NUM_FUEL_HOLES = 52
    DIA_FUEL_HOLES = 0.034 * ureg.inch
    NUM_HEAD_FFC_HOLES = 32
    DIA_HEAD_FFC_HOLES = 0.0135 * ureg.inch
    Cd_PFM_PC = 0.78 # Determined empirically through waterflow data
    A_PFM_PC = np.pi / 4 * (NUM_FUEL_HOLES * DIA_FUEL_HOLES ** 2 + NUM_HEAD_FFC_HOLES * DIA_HEAD_FFC_HOLES ** 2)
    CdA_PFM_PC = Cd_PFM_PC * A_PFM_PC

    # w4 components
    # CdA_PFCI_PFCMT
    #   CdA of throat film coolant orifice and lines
    # CdA_PFCMT_PC
    #   CdA of chamber holes
    w4guess = w2guess * (PERCENT_FFC/100.0)
    D4 = 0.25 * ureg.inch
    Re4 = (4 / np.pi) * (w4guess / (D4.to('ft') * RP1_DynamicViscosity_DEFAULT))
    f4 = fl.friction_factor(Re4, eD=ABS_ROUGHNESS_SS/D4)

    Dia_ORFCT = 0.07 * ureg.inch
    CdA_PFCI_PFCMT = fl.flow_meter.C_Reader_Harris_Gallagher(D=D4.to_base_units().magnitude,
                                                             Do=Dia_ORFCT.to_base_units().magnitude,
                                                             rho=RP1_Density_DEFAULT.to_base_units().magnitude,
                                                             mu=RP1_DynamicViscosity_DEFAULT.to_base_units().magnitude,
                                                             m=w4guess.to_base_units().magnitude, taps='flange') * np.pi/4 * Dia_ORFCT ** 2

    NUM_THROAT_FFC_HOLES = 24
    DIA_THROAT_FFC_HOLES = 0.0135 * ureg.inch
    Cd_PFCMT_PC = 0.7
    CdA_PFCMT_PC = Cd_PFCMT_PC * (NUM_THROAT_FFC_HOLES * np.pi / 4 * DIA_THROAT_FFC_HOLES ** 2)

    # w5 components
    # CdA_PRO_POTI
    #   Lines leading to tank inlet, check valve
    # CdA_POTI_POT
    #   Diffuser tank section, portion of tank filled with gas
    # CdA_POT_POTO
    #   Liquid oxygen in tank to converging duct on outlet
    # CdA_POTO_PMOVI
    #   Lines leading up to main ox valve
    # CdA_PMOVI_PMOVO
    #   CdA of main oxidizer valve
    # CdA_PMOVO_POM
    #   CdA of lines leading to fuel manifold
    # CdA_POM_PC
    #   CdA of ox injector
    w5guess = 2.53 * (ureg.lb / ureg.s)
    D5 = 0.5 * ureg.inch
    Re5 = (4 / np.pi) * (w5guess / (D5.to('ft') * LOX_DynamicViscosity_DEFAULT))
    f5 = fl.friction_factor(Re5, eD=ABS_ROUGHNESS_SS/D5)

    Tank_Diameter = 1 * ureg.ft
    K_POT_POTO = fl.fittings.contraction_conical(Tank_Diameter.to('m'), D2, f2, l=4*ureg.inch)
    CdA_POT_POTO = K_to_CdA(K_POT_POTO,  D=D2)

    K_POTO_PMOVI = fl.K_from_f(f5, L=(5*ureg.ft).to('in'), D=D5) # line losses
    K_POTO_PMOVI += fl.fittings.bend_rounded(D5, angle=90, fd=f5, bend_diameters=5)
    K_POTO_PMOVI += fl.fittings.bend_rounded(D5, angle=90, fd=f5, bend_diameters=5)
    CdA_POTO_PMOVI = K_to_CdA(K_POTO_PMOVI, D=D5)

    K_PMOVI_PMOVO = fl.fittings.K_ball_valve_Crane(D1=D5,D2=D5, angle=0, fd=f5)
    CdA_PMOVI_PMOVO = K_to_CdA(K_PMOVI_PMOVO, D=D5)

    K_PMOVO_POM = fl.K_from_f(f5, L=(3*ureg.ft).to('in'), D=D5)
    CdA_PMOVO_POM = K_to_CdA(K_PMOVO_POM, D=D5)


    # ARES 2017-2018 INJECTOR GEOMETRIES
    # NUM_OX_HOLES = 16
    # DIA_OX_HOLES = 0.070 * ureg.inch
    # Cd_POM_PC = 0.685 # Empirical waterflow testing of injector ares 2017-2018
    # CdA_POM_PC = NUM_OX_HOLES * np.pi/4 * DIA_OX_HOLES ** 2 * Cd_POM_PC

    # BPL 2018-2019 INJECTOR GEOMETRIES
    NUM_OX_HOLES = 24
    DIA_OX_HOLES = 0.067 * ureg.inch
    Cd_POM_PC = 0.685 # Empirical waterflow testing of injector ares 2017-2018
    CdA_POM_PC = NUM_OX_HOLES * np.pi/4 * DIA_OX_HOLES ** 2 * Cd_POM_PC

    # Get leg alpha CdA's
    # Recall, alpha = (12 / CdA ) ** 2 * 1 / (2*g*rho)
    a2cda = CdA_sum_series([CdA_PFT_PFTO, CdA_PFTO_PMFVI,CdA_PMFVI_PMFVO, CdA_PMFVO_PFCI])
    a3cda = CdA_sum_series([CdA_PFCI_PFM, CdA_PFM_PC])
    a4cda = CdA_sum_series([CdA_PFCI_PFCMT, CdA_PFCMT_PC])
    a5cda = CdA_sum_series([CdA_POT_POTO, CdA_POTO_PMOVI, CdA_PMOVI_PMOVO, CdA_PMOVO_POM, CdA_POM_PC])

    a2 = (12 / a2cda) ** 2 * 1 / (2 * G_const * RP1_Density_DEFAULT)
    a3 = (12 / a3cda) ** 2 * 1 / (2 * G_const * RP1_Density_DEFAULT)
    a4 = (12 / a4cda) ** 2 * 1 / (2 * G_const * RP1_Density_DEFAULT)
    a5 = (12 / a5cda) ** 2 * 1 / (2 * G_const * LOX_Density_DEFAULT)

    w1guess = w2guess + w5guess
    PCguess = 300 * ureg.psi
    guessarr = (w1guess.magnitude, w2guess.magnitude, w3guess.magnitude, w4guess.magnitude, w5guess.magnitude, PCguess.magnitude)
    PORO = 360 * ureg.psi
    PFRO = 520 * ureg.psi
    DIA_THROAT = 1.658 * ureg.inch
    At = np.pi / 4 * DIA_THROAT ** 2
    cstar = (1805 * ureg.m / ureg.s).to('ft / s') * 0.7

    data = (a2.magnitude, a3.magnitude, a4.magnitude, a5.magnitude, PORO.magnitude, PFRO.magnitude, At.magnitude)
    sol = sp.fsolve(equations, guessarr, args=data)
    w1, w2, w3, w4, w5, PC = sol
    print('Total mdot: '+str(w1 * ureg.lb / ureg.s))
    print('Total fuel mdot: '+str(w2 * ureg.lb / ureg.s))
    print('Total Ox mdot: '+str(w5 * ureg.lb / ureg.s))
    print('Mixture Ratio: '+str(w5/w2))
    print('Total throat FFC mdot: '+str(w4 * ureg.lb / ureg.s))
    print('Percent throat FFC: '+str(w4/w2*100))
    print('Chamber Pressure: '+str(PC * ureg.psi))
    print('Cstar: '+str(getCstar(w5, w2) * ureg.m / ureg.s))
    print('Thrust: '+str((PC * ureg.psi * At).to('lbf')))
    print('Done!')

def equations(p, *data):
    a2, a3, a4, a5, PORO, PFRO, At = data
    w1, w2, w3, w4, w5, PC = p
    return (
            # Mass conservation equations
            w1 - w2 - w5,
            w2 - w3 - w4,
            # Energy conservation equations
            -a3*w3 ** 2 + a4*w4 ** 2,
            PORO - PC - a5*w5 ** 2,
            PFRO - PC - (a3 * w3 ** 2 + a2 * w2 ** 2),
            # PC dependence on mdot
            PC * At - w1 / G_const * getCstar(w5, w2) * 3.28
    )

def getCstar(mdot_ox, mdot_fuel):
    OF = mdot_ox / mdot_fuel
    s = InterpolatedUnivariateSpline(RP1_PROPERTIES['OF'], RP1_PROPERTIES['cstar'], k=2)
    return s(OF).tolist()

def getHydraulicDiameterRectangle(a, b):
    return 2 * a * b / (a + b)

def K_to_CdA(K, D):
    return fl.K_to_Cv(K, D) / 38.0

def CdA_sum_series(CdA_arr):
    denom = 0
    for CdA in CdA_arr:
        denom += 1 / (CdA ** 2)
    return 1 / np.sqrt(denom)

if __name__ == '__main__':
    run()