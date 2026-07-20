"""
Get the Cp, H, and S from NASA9

Prerequisite: ./table/thermo.inp
Credit: McBride, B.J., Zehe, M.J., and Gordon, S., 
"NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species,"
"""
import numpy as np

R = 8.31446261815324
J_PER_EV_MOL = 96485.33212

ALIASES = {
    "OCS": "COS",
    "E": "E-", # electron
}

EXCITED_STATES = { #[eV]
    "O_1D": ("O", 1.967),
    "O_1S": ("O", 4.189),
    "N_2D": ("N", 2.384),
    "N_2P": ("N", 3.576),
    "O+_2D": ("O+", 3.324),
    "O+_2P": ("O+", 5.017),
    "O2_DG": ("O2", 0.9773),
    "O2_SG": ("O2", 1.6268),
    "CH2_SG": ("CH2", 0.392),
}


def clean_name(name):
    name = name.strip().upper()
    if name in EXCITED_STATES:
        return EXCITED_STATES[name][0]
    if name in ALIASES:
        return ALIASES[name]
    return name.split(",")[0]


def excitation_energy(name):
    name = name.strip().upper()
    if name in EXCITED_STATES:
        return EXCITED_STATES[name][1] * J_PER_EV_MOL
    return 0.0


def name_matches(name, species):
    if name is None:
        return False
    return clean_name(name) == clean_name(species)


def get_coefficients(species, T, verbose=False):
    """
    @ input: 
    - species: chemical species e.g. "CH4"/"Ar+"
    - T: temperature of the reaction [K]

    @ Output:
    - list of 9 coefficients [a1, a2, ..., a7, b1, b2]
    """

    with open("./table/thermo.inp", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        coefficients = []

        for i, line in enumerate(lines):
            # find the species
            name = line.split()[0] if line.strip() and not line.startswith(("!", "#")) else None
            if not name_matches(name, species):
                continue

            meta = lines[i+1].split()
            temp_interval = meta[0]  # number of temperature intervals

            if verbose:
                print(name)
                print(f"Phase: {meta[-3]} (Gas: 0, Condensed: non-zero)")
                print(f"Molecular weight: {meta[-2]}")
                print(f"ΔH°(298.15 K) in J/mol: {meta[-1]}")

            for interval in range(int(temp_interval)):
                temp_line = lines[i+1+3*interval+1]  # fixed-width fields

                t_low = float(temp_line[0:11])
                t_high = float(temp_line[11:22])

                if verbose:
                    print(f"interval {interval}: {t_low} - {t_high} K")

                if t_low <= T <= t_high:

                    coeff_line1 = lines[i+1+3*interval+2].replace('D', 'E') # each temperature interval block have three lines
                    coeff_line2 = lines[i+1+3*interval+3].replace('D', 'E')

                    nums1 = [coeff_line1[j:j+16] for j in range(0, 80, 16)] # each column is 16 character size
                    nums2 = [coeff_line2[j:j+16] for j in range(0, 80, 16)]
                    nums = [float(x) for x in (nums1 + nums2) if x.strip()]

                    coefficients = nums  # a1 - a7, b1, b2 
                    break

            if len(coefficients) == 0:
                raise ValueError("Temperature does not exist in the database interval.")

            return coefficients

    raise ValueError(f"Species '{species}' not found")

def get_Cp(species, T):
    """
    Return Cp [J/K-mol]
    """
    coefficients = get_coefficients(species=species, T=T)
    exp = [-2, -1, 0, 1, 2, 3, 4]
    CpR = sum(coefficients[i] * T ** exp[i] for i in range(7)) # Cp/R
    return CpR * R

def get_H(species, T):
    """
    Return H [J/mol]
    """   
    coefficients = get_coefficients(species=species, T=T)
    HRT = 0 # H/RT
    HRT -= coefficients[0] / T**2
    HRT += coefficients[1] * np.log(T) / T
    HRT += coefficients[2]                          

    for i in range(3, 7):
        power = i - 2
        HRT += coefficients[i] * T**power / (power + 1)
    HRT += coefficients[7] / T                       

    return HRT * R * T + excitation_energy(species)

def get_S(species, T, p=1.0e5):
    """
    Return S [J/K-mol]
    p: pressure [Pa], see S_trans in partition.py
    T: temperature [K]
    """
    coefficients = get_coefficients(species=species, T=T)

    SR = 0 # S/R

    SR -= coefficients[0] * T**-2 / 2
    SR -= coefficients[1] / T
    SR += coefficients[2] * np.log(T)
    SR += np.log(1.0e5/p)

    for i in range(3, 7):
        power = i - 2
        SR += coefficients[i] * T**power / power

    SR += coefficients[8]    

    return SR * R
