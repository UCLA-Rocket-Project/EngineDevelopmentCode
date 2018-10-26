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
HE_DynamicViscosity_DEFAULT = (2.10E-5 * ureg.Pa * ureg.s).to('lb / (ft * s)')
# Source: https://www.sciencedirect.com/science/article/pii/S0011227507001506

RP1_Density_DEFAULT = 51.4 * ureg.lb / (ureg.ft ** 3)
LOX_Density_DEFAULT = 75.038409 * ureg.lb / (ureg.ft ** 3)
# Source: https://www.sciencedirect.com/science/article/pii/S0011227507001506

ABS_ROUGHNESS_ALUMINUM = (6.56E-6 * ureg.feet).to('inch')# Source: https://www.engineeringtoolbox.com/surface-roughness-ventilation-ducts-d_209.html
ABS_ROUGHNESS_SS = (0.0197E-3 * ureg.inch).to('inch')
PERCENT_FFC = 11.5 # percent fuel film coolant
G_const = 32.174

w_fu = 7
w_ox = 15.4

# Tubing geometries
t2 = 0.049
D2_OD = 1.0
t5 = 0.049
D5_OD = 1.0
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
    w2guess = w_fu * (ureg.lb / ureg.s)

    D2 = (D2_OD - 2*t2) * ureg.inch
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

    K_PMFVI_PMFVO = fl.fittings.K_ball_valve_Crane(D1=D2*0.8,D2=D2, angle=0, fd=f2)
    CdA_PMFVI_PMFVO = K_to_CdA(K_PMFVI_PMFVO, D=D2)

    K_PMFVO_PREGI = fl.K_from_f(f2, L=(3.5*ureg.ft).to('in'), D=D2)
    K_PMFVO_PREGI += fl.fittings.bend_rounded(D2, angle=90, fd=f2, bend_diameters=5)
    K_PMFVO_PREGI += fl.fittings.bend_rounded(D2, angle=90, fd=f2, bend_diameters=5)
    CdA_PMFVO_PREGI = K_to_CdA(K_PMFVO_PREGI, D=D2)

    # PITCH_NOZZLE_OUTLET = CHANNEL_WIDTH_NOZZLE_OUTLET * np.sin(np.deg2rad(EXPANSION_ANGLE))
    # NUMCOILS_NOZZLE = (R_NOZZLE_EXIT - (R_THROAT + R_NOZZLE_EXIT * 0.3)) / PITCH_NOZZLE_OUTLET
    # PITCH_NOZZLE_OUTLET = (R_NOZZLE_EXIT - (R_THROAT + RDIFF)) / NUMCOILS_NOZZLE

    # Nozzle "spiral"
    THROAT_COOLING_VELOCITY = 40 * ureg.ft / ureg.s
    NOZZLE_CHAMBER_COOLING_VELOCITY =  20 * ureg.ft / ureg.s
    COOLING_CHANNEL_HEIGHT = 0.3 * ureg.inch
    CHANNEL_WIDTH_NOZZLE = (w2guess / (RP1_Density_DEFAULT * NOZZLE_CHAMBER_COOLING_VELOCITY * COOLING_CHANNEL_HEIGHT)).to('in')
    CHANNEL_WIDTH_THROAT = (w2guess / (RP1_Density_DEFAULT * THROAT_COOLING_VELOCITY * COOLING_CHANNEL_HEIGHT)).to('in')
    # Spacing between each coil
    CHANNEL_WIDTH_HEAD = CHANNEL_WIDTH_NOZZLE

    CHAMBER_THICKNESS = 0.15 * ureg.inch
    CHAMBER_PRESSURE = 300 * ureg.psi
    mdot_total = (w_fu + w_ox) * ureg.lb/ureg.s
    cstar = (getCstar(w_ox, w_fu) * ureg.m / ureg.s).to('ft / s')
    AREA_THROAT = ((mdot_total * cstar) / CHAMBER_PRESSURE).to('in ** 2')
    ID_THROAT = np.sqrt(4 * AREA_THROAT / np.pi)
    EXPANSION_RATIO = 4.15
    CONTRACTION_RATIO = 6
    DIA_THROAT = ID_THROAT + 2 * CHAMBER_THICKNESS
    R_THROAT = DIA_THROAT / 2
    AREA_NOZZLE_EXIT = AREA_THROAT * EXPANSION_RATIO
    DIA_NOZZLE_EXIT = np.sqrt(4 * AREA_NOZZLE_EXIT / np.pi) + 2 * CHAMBER_THICKNESS
    R_NOZZLE_EXIT = DIA_NOZZLE_EXIT/2

    # Radius where channel width is decreased in order to increase fluid velocity and provide more cooling
    # In reality this is a gradual change, but for simplicity an average width will be taken with an abrupt change
    R_WIDTHCHANGE = R_THROAT + (R_NOZZLE_EXIT - R_THROAT) * 0.5
    AREA_CHAMBER = AREA_THROAT * CONTRACTION_RATIO
    DIA_CHAMBER = np.sqrt(4 * AREA_CHAMBER / np.pi ) + 2 * CHAMBER_THICKNESS
    R_CHAMBER = DIA_CHAMBER / 2

    DH_NOZZLE = getHydraulicDiameterRectangle(COOLING_CHANNEL_HEIGHT, CHANNEL_WIDTH_NOZZLE) # Hydraulic Diameter, approx
    DH_THROAT = getHydraulicDiameterRectangle(COOLING_CHANNEL_HEIGHT, CHANNEL_WIDTH_THROAT)
    DH_HEAD = DH_NOZZLE

    EXPANSION_ANGLE = 11
    CONTRACTION_ANGLE = 45
    PITCH_NOZZLE = CHANNEL_WIDTH_NOZZLE * np.cos(np.deg2rad(EXPANSION_ANGLE))
    PITCH_THROAT_EXPANSION = CHANNEL_WIDTH_THROAT * np.cos(np.deg2rad(EXPANSION_ANGLE))
    PITCH_THROAT_CONTRACTION = CHANNEL_WIDTH_THROAT * np.cos(np.deg2rad(CONTRACTION_ANGLE))
    PITCH_HEAD = CHANNEL_WIDTH_HEAD

    NUMCOILS_NOZZLE = (2 * ureg.inch) / PITCH_NOZZLE
    NUMCOILS_THROAT_EXPANSION = (2 * ureg.inch) / PITCH_THROAT_EXPANSION
    NUMCOILS_THROAT_CONTRACTION = (0.8 * ureg.inch) / PITCH_THROAT_CONTRACTION
    NUMCOILS_HEAD = (5.4 * ureg.inch) / PITCH_HEAD

    # Loss through diverging nozzle section
    Re_NOZZLE = (4 / np.pi) * (w2guess / (DH_NOZZLE.to('ft') * RP1_DynamicViscosity_DEFAULT))
    f_NOZZLE = fl.friction_factor(Re_NOZZLE, eD=ABS_ROUGHNESS_SS/DH_NOZZLE)
    K_NOZZLE = fl.fittings.helix(DH_NOZZLE, (R_NOZZLE_EXIT + R_WIDTHCHANGE)/2, PITCH_NOZZLE, NUMCOILS_NOZZLE, f_NOZZLE)
    CdA_NOZZLE_REGEN = K_to_CdA(K_NOZZLE, DH_NOZZLE)

    # Loss through throat regen, diverging section
    Re_THROAT = (4 / np.pi) * (w2guess / (DH_THROAT.to('ft') * RP1_DynamicViscosity_DEFAULT))
    f_THROAT = fl.friction_factor(Re_THROAT, eD=ABS_ROUGHNESS_SS/DH_THROAT)
    K_THROAT_EXPANSION = fl.fittings.helix(DH_THROAT, (R_WIDTHCHANGE + R_THROAT)/2, PITCH_THROAT_EXPANSION,
                                     NUMCOILS_THROAT_EXPANSION, f_THROAT)
    CdA_THROAT_EXPANSION = K_to_CdA(K_THROAT_EXPANSION, DH_THROAT)

    # Loss through throat regen, contracting section
    K_THROAT_CONTRACTION = fl.fittings.helix(DH_THROAT, (R_CHAMBER + R_THROAT)/2, PITCH_THROAT_CONTRACTION,
                                     NUMCOILS_THROAT_CONTRACTION, f_THROAT)
    CdA_THROAT_CONTRACTION = K_to_CdA(K_THROAT_CONTRACTION, DH_THROAT)

    CdA_PREGI_PFCI = CdA_sum_series([CdA_NOZZLE_REGEN,
                                    CdA_THROAT_EXPANSION,
                                    CdA_THROAT_CONTRACTION])

    # w3 components
    # CdA_PFCI_PFM
    #   Regen circuit leading to fuel manifold
    # CdA_PFM_PC
    #   CdA of fuel injector
    w3guess = w2guess * ((100-PERCENT_FFC)/100.0)
    Re_HEAD = (4 / np.pi) * (w3guess / (DH_HEAD.to('ft') * RP1_DynamicViscosity_DEFAULT))
    f3 = fl.friction_factor(Re_HEAD, eD=ABS_ROUGHNESS_SS/DH_HEAD)
    K_PFCI_PFM = fl.fittings.helix(DH_HEAD, R_CHAMBER, CHANNEL_WIDTH_HEAD, NUMCOILS_HEAD, f3)
    CdA_PFCI_PFM = K_to_CdA(K_PFCI_PFM, D=DH_HEAD)

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
    INJECTOR_PRESSURE_DROP = CHAMBER_PRESSURE*0.26
    Cd_PFM_PC = 0.7
    A_FUEL_HOLES = ((12 * w2guess / Cd_PFM_PC) / np.sqrt(2 * G_const * RP1_Density_DEFAULT * INJECTOR_PRESSURE_DROP)).magnitude * ureg.inch ** 2
    NUM_FUEL_HOLES = 150
    DIA_FUEL_HOLES = np.sqrt(4 / np.pi * A_FUEL_HOLES/NUM_FUEL_HOLES)
    # NUM_HEAD_FFC_HOLES = 32
    # DIA_HEAD_FFC_HOLES = 0.0135 * ureg.inch

    # A_HEAD_FFC_HOLES = np.pi / 4 * NUM_HEAD_FFC_HOLES * DIA_HEAD_FFC_HOLES ** 2
    # CdA_HEAD_FFC_HOLES = Cd_PFM_PC * A_HEAD_FFC_HOLES
    CdA_FUEL_HOLES = Cd_PFM_PC * A_FUEL_HOLES
    CdA_PFM_PC = CdA_FUEL_HOLES

    # w4 components
    # CdA_PFCI_PC
    #   CdA of chamber holes
    w4guess = w2guess * (PERCENT_FFC/100.0)
    Cd_PFCI_PC = 0.7
    THROAT_FFC_VELOCITY = 52 * ureg.ft / ureg.s
    THROAT_FFC_DP = (144 * THROAT_FFC_VELOCITY.magnitude ** 2)/(2 * G_const * RP1_Density_DEFAULT.magnitude)

    # D4 = 0.25 * ureg.inch
    # Re4 = (4 / np.pi) * (w4guess / (DH_THROAT.to('ft') * RP1_DynamicViscosity_DEFAULT) )
    # f4 = fl.friction_factor(Re4, eD=ABS_ROUGHNESS_SS/DH_THROAT)
    A_THROAT_FFC_HOLES = ((12 * w4guess / Cd_PFCI_PC) / np.sqrt(
        2 * G_const * RP1_Density_DEFAULT * THROAT_FFC_DP)).magnitude * ureg.inch ** 2
    NUM_THROAT_FFC_HOLES = 24
    DIA_FUEL_HOLES = np.sqrt(4 / np.pi * A_THROAT_FFC_HOLES / NUM_THROAT_FFC_HOLES)

    CdA_PFCI_PC = Cd_PFCI_PC * A_THROAT_FFC_HOLES

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
    w5guess = w_ox * (ureg.lb / ureg.s)

    D5 =  (D5_OD - 2*t5) * ureg.inch
    Re5 = (4 / np.pi) * (w5guess / (D5.to('ft') * LOX_DynamicViscosity_DEFAULT))
    f5 = fl.friction_factor(Re5, eD=ABS_ROUGHNESS_SS/D5)

    Tank_Diameter = 1 * ureg.ft
    K_POT_POTO = fl.fittings.contraction_conical(Tank_Diameter.to('m'), D2, f2, l=4*ureg.inch)
    CdA_POT_POTO = K_to_CdA(K_POT_POTO,  D=D2)

    K_POTO_PMOVI = fl.K_from_f(f5, L=(5*ureg.ft).to('in'), D=D5) # line losses
    K_POTO_PMOVI += fl.fittings.bend_rounded(D5, angle=90, fd=f5, bend_diameters=5)
    K_POTO_PMOVI += fl.fittings.bend_rounded(D5, angle=90, fd=f5, bend_diameters=5)
    CdA_POTO_PMOVI = K_to_CdA(K_POTO_PMOVI, D=D5)

    K_PMOVI_PMOVO = fl.fittings.K_ball_valve_Crane(D1=D5*0.8,D2=D5, angle=0, fd=f5)
    CdA_PMOVI_PMOVO = K_to_CdA(K_PMOVI_PMOVO, D=D5)

    K_PMOVO_POM = fl.K_from_f(f5, L=(3*ureg.ft).to('in'), D=D5)
    CdA_PMOVO_POM = K_to_CdA(K_PMOVO_POM, D=D5)

    # ARES 2017-2018 INJECTOR GEOMETRIES
    # NUM_OX_HOLES = 16
    # DIA_OX_HOLES = 0.070 * ureg.inch
    # Cd_POM_PC = 0.685 # Empirical waterflow testing of injector ares 2017-2018
    # CdA_POM_PC = NUM_OX_HOLES * np.pi/4 * DIA_OX_HOLES ** 2 * Cd_POM_PC

    # BPL 2018-2019 INJECTOR GEOMETRIES

    Cd_POM_PC = 0.7
    A_OX_HOLES = ((12 * w5guess / Cd_POM_PC) / np.sqrt(2 * G_const * LOX_Density_DEFAULT * INJECTOR_PRESSURE_DROP)).magnitude * ureg.inch ** 2
    NUM_OX_HOLES = 150
    DIA_OX_HOLES = np.sqrt(4 / np.pi * A_OX_HOLES/NUM_OX_HOLES)
    CdA_POM_PC = Cd_POM_PC * A_OX_HOLES

    # Get leg alpha CdA's
    # Recall, alpha = (12 / CdA ) ** 2 * 1 / (2*g*rho)
    a2cda = CdA_sum_series([CdA_PFT_PFTO, CdA_PFTO_PMFVI,CdA_PMFVI_PMFVO, CdA_PMFVO_PREGI, CdA_PREGI_PFCI])
    a3cda = CdA_sum_series([CdA_PFCI_PFM, CdA_PFM_PC])
    a4cda = CdA_PFCI_PC
    CdA_PFCI_PC_Total = a3cda + a4cda
    CdA_PREGI_PC = CdA_sum_series([CdA_PFCI_PC_Total, CdA_PREGI_PFCI])
    a5cda = CdA_sum_series([CdA_POT_POTO, CdA_POTO_PMOVI, CdA_PMOVI_PMOVO, CdA_PMOVO_POM, CdA_POM_PC])

    a2 = (12 / a2cda) ** 2 * 1 / (2 * G_const * RP1_Density_DEFAULT)
    a3 = (12 / a3cda) ** 2 * 1 / (2 * G_const * RP1_Density_DEFAULT)
    a4 = (12 / a4cda) ** 2 * 1 / (2 * G_const * RP1_Density_DEFAULT)
    a5 = (12 / a5cda) ** 2 * 1 / (2 * G_const * LOX_Density_DEFAULT)

    # Estimated DP's (based on flow rate guesses/targets)
    print('Target DP\'s based on CdA, flow rate')
    DP_POTO_PMOVIg = CdA_to_DP(CdA_POTO_PMOVI, w5guess, LOX_Density_DEFAULT)
    DP_PMOVI_PMOVOg = CdA_to_DP(CdA_PMOVI_PMOVO, w5guess, LOX_Density_DEFAULT)
    DP_PMOVO_POMg = CdA_to_DP(CdA_PMOVO_POM, w5guess, LOX_Density_DEFAULT)
    DP_POM_PCg = CdA_to_DP(CdA_POM_PC, w5guess, LOX_Density_DEFAULT)
    DP_OX_LINES = DP_POTO_PMOVIg + DP_PMOVI_PMOVOg + DP_PMOVO_POMg

    DP_PFTO_PMFVIg = CdA_to_DP(CdA_PFTO_PMFVI, w2guess, RP1_Density_DEFAULT)
    DP_PMFVI_PMFVOg = CdA_to_DP(CdA_PMFVI_PMFVO, w2guess, RP1_Density_DEFAULT)
    DP_PMFVO_PREGIg = CdA_to_DP(CdA_PMFVO_PREGI, w2guess, RP1_Density_DEFAULT)
    DP_PREGI_PFCIg = CdA_to_DP(CdA_PREGI_PFCI, w2guess, RP1_Density_DEFAULT)
    DP_PFCI_PCg = CdA_to_DP(CdA_PFCI_PC, w4guess, RP1_Density_DEFAULT)
    DP_PFCI_PFMg = CdA_to_DP(CdA_PFCI_PFM, w3guess, RP1_Density_DEFAULT)
    DP_PFM_PCg = CdA_to_DP(CdA_PFM_PC, w3guess, RP1_Density_DEFAULT)
    DP_FUEL_LINES = DP_PFTO_PMFVIg+DP_PMFVI_PMFVOg+DP_PMFVO_PREGIg

    w1guess = w2guess + w5guess
    PCguess = 300 * ureg.psi
    guessarr = (w1guess.magnitude, w2guess.magnitude, w3guess.magnitude, w4guess.magnitude, w5guess.magnitude, PCguess.magnitude)
    PORO_guess = PCguess.magnitude + (a5 * w5guess ** 2).magnitude
    PFRO_guess = PCguess.magnitude + (a3 * w3guess ** 2).magnitude + (a2 * w2guess ** 2).magnitude
    PORO = 458 * ureg.psi
    PFRO = 562 * ureg.psi
    DIA_THROAT = ID_THROAT
    At = np.pi / 4 * DIA_THROAT ** 2

    # cstar = (1805 * ureg.m / ureg.s).to('ft / s')
    cstarEffciency = 0.95
    ct = 1.4 # thrust coefficient
    data = (a2.magnitude, a3.magnitude, a4.magnitude, a5.magnitude, PORO.magnitude, PFRO.magnitude, At.magnitude, cstarEffciency)
    sol = sp.fsolve(equations, guessarr, args=data)
    w1, w2, w3, w4, w5, PC = sol
    w3_main = w3 * CdA_FUEL_HOLES / CdA_PFM_PC
    # w3_head_ffc = w3 * CdA_HEAD_FFC_HOLES / CdA_PFM_PC
    DP_PREGI_PFM = CdA_to_DP(CdA_PREGI_PFCI, w2, rho=RP1_Density_DEFAULT) + CdA_to_DP(CdA_PFCI_PFM, w3, rho=RP1_Density_DEFAULT)
    DP_PFM_PC = CdA_to_DP(CdA_PFM_PC, w3, rho=RP1_Density_DEFAULT)
    DP_POM_PC = CdA_to_DP(CdA_POM_PC, w5, rho=LOX_Density_DEFAULT)

    print('Total mdot: '+str(w1 * ureg.lb / ureg.s))
    print('Total fuel mdot: '+str(w2 * ureg.lb / ureg.s))
    print('Total Ox mdot: '+str(w5 * ureg.lb / ureg.s))
    print('Mixture Ratio: '+str(w5/w2))
    # print('Total throat FFC mdot: '+str(w4 * ureg.lb / ureg.s))
    print('Percent throat FFC: '+str(w4/w2*100))
    # print('Percent head FFC: '+str(w3_head_ffc / w2 * 100))
    print('Regen pressure drop: '+str(DP_PREGI_PFM))
    print('Fuel Injector Pressure Drop: '+str(DP_PFM_PC))
    print('Ox Injector Pressure Drop: '+str(DP_POM_PC))
    print('Chamber Pressure: '+str(PC * ureg.psi))
    print('Cstar: '+str(getCstar(w5, w2) * ureg.m / ureg.s))
    print('Thrust: '+str((PC * ureg.psi * At * ct).to('lbf')))
    print('Done!')

def equations(p, *data):
    a2, a3, a4, a5, PORO, PFRO, At, cstarEfficiency = data
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
            PC * At - w1 / G_const * getCstar(w5, w2) * cstarEfficiency * 3.28
    )

def getCstar(mdot_ox, mdot_fuel):
    OF = mdot_ox / mdot_fuel
    s = InterpolatedUnivariateSpline(RP1_PROPERTIES['OF'], RP1_PROPERTIES['cstar'], k=2)
    return s(OF).tolist()

def getHydraulicDiameterRectangle(a, b):
    return 2 * a * b / (a + b)

# def K_to_CdA(K, D):
#     return fl.K_to_Cv(K, D) / 38.0

def K_to_CdA(K, D):
    return 1 / np.sqrt(K) * np.pi / 4 * D ** 2

def CdA_sum_series(CdA_arr):
    denom = 0
    for CdA in CdA_arr:
        denom += 1 / (CdA ** 2)
    return 1 / np.sqrt(denom)

def CdA_to_DP(CdA, w, rho):
    return (12 * magnitude(w) / magnitude(CdA)) ** 2 * 1 / (2 * G_const * magnitude(rho))

def magnitude(val):
    try:
        x = val.magnitude
    except:
        x = val
    return x

if __name__ == '__main__':
    run()