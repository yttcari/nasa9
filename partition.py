"""
Fit the nasa9 coefficient from table/exotic_vib.dat
File IO parts are written by codex
"""
import re
import numpy as np
import matplotlib.pyplot as plt

R = 8.31446261815324          # J / (mol K)
C2 = 1.4387768775039337       # cm*K  (hc/k)
kB = 1.380649e-23             # J/K
h = 6.62607015e-34            # J s
NA = 6.02214076e23            # 1/mol
ATM = 1.0e5                   # 1 atm in Par
me = 5.48579909065e-4         # g/mol
c = 3e8 * 100                 # cm/s
Hartree2J = 4.3597e-18        # J

ATOMIC_MASS = {
    "H": 1.008, "He": 4.0026, 
    "C": 12.011, "N": 14.007, "O": 15.999, 
    "S": 32.06, "Cl": 35.45, "Ar": 39.95, }

THERMO_PATH = "./table/thermo.inp"

def get_mass(name):
    charge = name.count("+") - name.count("-")
    core = name.rstrip("+-")
    mass = 0.0
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", core):
        if not sym:
            continue
        if sym not in ATOMIC_MASS:
            raise ValueError(f"unknown element '{sym}' in formula '{name}'")
        mass += ATOMIC_MASS[sym] * (int(num) if num else 1)
    if mass == 0.0:
        raise ValueError(f"could not parse formula '{name}'")
    return mass, charge


def parse_data_file(path):
    """
    @ Output
    Returns {name: {"sigma", "freq", "elec", "rot", "mass", "charge"}}.
    """
    with open(path) as f:
        raw = [ln.strip() for ln in f]

    # split into blocks on species
    blocks, current = [], None
    for ln in raw:
        if not ln or ln.startswith("#"):
            continue
        if ln[0].isalpha():
            current = (ln, [])
            blocks.append(current)
        elif current is not None:
            current[1].append(ln)
        else:
            raise ValueError(f"data line before any species name: '{ln}'")

    species = {}
    for name, lines in blocks:
        print(f"Processing {name}")

        sigma = int(float(lines[0]))
        freqs = np.array([float(t) for t in lines[1].split()])
        rot = np.array([float(t) for t in lines[-2].split()])
        elec_lines = lines[2:-2]
        E0 = float(lines[-1])
        elec = []
        for ln in elec_lines:
            toks = ln.split()
            print(toks)
            
            elec.append({"deg": float(toks[0]), "E": float(toks[1])}) # E [cm -1]
        

        mass, charge = get_mass(name)
        species[name] = {"sigma": sigma, "freq": freqs, "elec": elec,
                         "rot": rot, "mass": mass, "charge": charge, "E0": E0}
    return species



# Translational
def S_trans(T, mass_gmol, p=ATM):
    m = mass_gmol * 1e-3 / NA                       # kg / molecule
    term1 = 3/2 * np.log(2.0 * np.pi * m / h**2)
    term2 = 5/2 * np.log(kB * T)
    term3 = -np.log(p)
    term4 = 5/2
    return R * (term1 + term2 + term3 + term4)


def Cp_trans():
    return 5/2 * R

def H_trans(T):
    return 5/2 * R * T


# Rotational

def get_I(A_cm, B_cm, C_cm):
    # I = h / (8*pi^2*c*B)
    IA = h / (8.0 * np.pi**2 * c * A_cm)
    IB = h / (8.0 * np.pi**2 * c * B_cm)
    IC = h / (8.0 * np.pi**2 * c * C_cm)
    return IA, IB, IC


def S_rot(T, sigma, rot_const, linear):
    if linear:
        return R * (np.log(kB * T / (sigma * h * c * rot_const[0])) + 1.0)
    else: 
        IA, IB, IC = get_I(rot_const[0], rot_const[1], rot_const[2])
        term1 = np.log(8.0 * np.pi**2 / sigma)
        term2 = 3/2 * np.log(2.0 * np.pi * kB * T / h**2)
        term3 = 0.5 * np.log(IA * IB * IC)
        term4 = 3/2
        return R * (term1 + term2 + term3 + term4)


def Cp_rot(linear):
    if linear:
        return R
    else:
        return 3/2 * R


def H_rot(T, linear):
    if linear:
        return R * T
    else:
        return 3/2 * R * T

# Vibrational
def S_vib(T, nu_cm):
    x = h * (c * nu_cm) / (kB * T)                        
    term1 = -R * np.log(1.0 - np.exp(-x))
    term2 = R * x * np.exp(-x) / (1.0 - np.exp(-x))
    return np.sum(term1) + np.sum(term2)


def Cp_vib(T, nu_cm):
    x = h * (c * nu_cm) / (kB * T)          
    return R * np.sum(x ** 2 * np.exp(-x) / (1.0 - np.exp(-x)) ** 2)


def H_vib(T, nu_cm):
    x = h * (c * nu_cm) / (kB * T)          
    return R * T * np.sum(x * np.exp(-x) / (1.0 - (np.exp(-x))))

# electronic

def S_elec(T, elec_states):
    sum0 = 0.0
    sum1 = 0.0
    for s in elec_states:
        x = h * c * s["E"] / (kB * T)
        tmp = s["deg"] * np.exp(-x)               
        sum0 += tmp
        sum1 += tmp * x
    return R * np.log(sum0) + R * sum1 / sum0


def Cp_elec(T, elec_states):
    sum0 = 0.0
    sum1 = 0.0
    sum2 = 0.0

    for s in elec_states:
        x = h * c * s["E"] / (kB * T)
        tmp = s["deg"] * np.exp(-x)
        sum0 += tmp
        sum1 += tmp * x
        sum2 += tmp * (x ** 2)

    return R * (sum2 / sum0) - R * (sum1 / sum0) ** 2


def H_elec(T, elec_states):
    sum0 = 0
    sum1 = 0

    for s in elec_states:
        x = h * c * s["E"] / (kB * T)
        tmp = s["deg"] * np.exp(-x)
        sum0 += tmp
        sum1 += tmp * x

    return R * T * sum1 / sum0


def total_thermo(T, sp):

    # translational
    Cp = Cp_trans()
    H = H_trans(T)
    S = S_trans(T, sp["mass"])

    # rotational
    rot = sp["rot"]
    if len(rot) == 1:
        linear = True
    else:
        linear = False
    
    Cp += Cp_rot(linear)
    H += H_rot(T, linear)
    S += S_rot(T=T, sigma=sp["sigma"], rot_const=rot, linear=linear)

    # vibrational
    Cp += Cp_vib(T, sp["freq"])
    H += H_vib(T, sp["freq"])
    S += S_vib(T, sp["freq"])

    # electronic
    Cp += Cp_elec(T, sp["elec"])
    H += H_elec(T, sp["elec"])
    S += S_elec(T, sp["elec"])

    # Adding Zero-point energy
    H += sp['E0'] * Hartree2J

    return Cp, H, S


def evaluate_nasa9(T, coeffs):
    T = np.asarray(T)
    coeffs = np.asarray(coeffs)

    a1, a2, a3, a4, a5, a6, a7, b1, b2 = coeffs
    cp = R * (a1/T**2 + a2/T + a3 + a4*T + a5*T**2 + a6*T**3 + a7*T**4)
    h = R * T * (-a1/T**2 + a2*np.log(T)/T + a3 + a4*T/2 + a5*T**2/3 + a6*T**3/4 + a7*T**4/5 + b1/T)
    s = R * (-a1/(2*T**2) - a2/T + a3*np.log(T) + a4*T + a5*T**2/2 + a6*T**3/3 + a7*T**4/4 + b2)

    return cp.item(), h.item(), s.item()

def fit_nasa9(sp, Tmin=200.0, Tmid=1000.0, Tmax=6000.0, npoints=300):
    def fit_range(T):
        cp, hrt, sr = [], [], []
        for i, Ti in enumerate(T):
            Cp, H, S = total_thermo(Ti, sp)
            cp.append(Cp / R)
            hrt.append(H / (R * Ti))
            sr.append(S / R)

        z = np.zeros_like(T)
        A_cp = np.column_stack([T**-2, T**-1, z+1, T, T**2, T**3, T**4, z, z])
        A_h = np.column_stack([-T**-2, np.log(T)/T, z+1, T/2, T**2/3, T**3/4, T**4/5, T**-1, z])
        A_s = np.column_stack([-0.5*T**-2, -T**-1, np.log(T), T, T**2/2, T**3/3, T**4/4, z, z+1])

        A = np.vstack([A_cp, A_h, A_s])
        y = np.concatenate([cp, hrt, sr])
        scale = np.linalg.norm(A, axis=0)
        coeffs, *_ = np.linalg.lstsq(A / scale, y, rcond=None)
        return coeffs / scale

    T_low = np.linspace(Tmin, Tmid, npoints)
    T_high = np.linspace(Tmid, Tmax, npoints)
    return fit_range(T_low), fit_range(T_high)

def plot_nasa9_fit(sp, Tmin=200.0, Tmid=1000.0, Tmax=6000.0, npoints=300):
    coeffs_low, coeffs_high = fit_nasa9(sp, Tmin, Tmid, Tmax, npoints)
    
    T_low = np.linspace(Tmin, Tmid, npoints)
    T_high = np.linspace(Tmid, Tmax, npoints)
    T_all = np.concatenate([T_low, T_high])
    
    cp_true, hrt_true, sr_true = [], [], []
    for Ti in T_all:
        Cp, H, S = total_thermo(Ti, sp)
        cp_true.append(Cp / R)
        hrt_true.append(H / (R * Ti))
        sr_true.append(S / R)
        
    def eval_nasa9(T, c):
        # cp/R = a1/T^2 + a2/T + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
        cp_fit = c[0]*T**-2 + c[1]*T**-1 + c[2] + c[3]*T + c[4]*T**2 + c[5]*T**3 + c[6]*T**4
        
        # h/RT = -a1/T^2 + a2*ln(T)/T + a3 + a4*T/2 + a5*T^2/3 + a6*T^3/4 + a7*T^4/5 + a8/T
        hrt_fit = -c[0]*T**-2 + c[1]*np.log(T)/T + c[2] + c[3]*T/2 + c[4]*T**2/3 + c[5]*T**3/4 + c[6]*T**4/5 + c[7]*T**-1
        
        # s/R = -a1/(2*T^2) - a2/T + a3*ln(T) + a4*T + a5*T^2/2 + a6*T^3/3 + a7*T^4/4 + a9
        sr_fit = -0.5*c[0]*T**-2 - c[1]*T**-1 + c[2]*np.log(T) + c[3]*T + c[4]*T**2/2 + c[5]*T**3/3 + c[6]*T**4/4 + c[8]
        
        return cp_fit, hrt_fit, sr_fit

    
    cp_fit_low, hrt_fit_low, sr_fit_low = eval_nasa9(T_low, coeffs_low)
    cp_fit_high, hrt_fit_high, sr_fit_high = eval_nasa9(T_high, coeffs_high)
    
    
    cp_fit = np.concatenate([cp_fit_low, cp_fit_high])
    hrt_fit = np.concatenate([hrt_fit_low, hrt_fit_high])
    sr_fit = np.concatenate([sr_fit_low, sr_fit_high])
    
    # 5. Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Cp/R Plot
    axs[0].plot(T_all, cp_true, 'k-', label='Partition Function', linewidth=2)
    axs[0].plot(T_all, cp_fit, 'r--', label='NASA 9 Fit', linewidth=2)
    axs[0].axvline(Tmid, color='gray', linestyle=':', label='T_mid split')
    axs[0].set_title('$C_p / R$ Fit Verification')
    axs[0].set_xlabel('Temperature (K)')
    axs[0].set_ylabel('$C_p / R$')
    axs[0].legend()
    
    # H/RT Plot
    axs[1].plot(T_all, hrt_true, 'k-', linewidth=2)
    axs[1].plot(T_all, hrt_fit, 'g--', linewidth=2)
    axs[1].axvline(Tmid, color='gray', linestyle=':')
    axs[1].set_title('$H / (RT)$ Fit Verification')
    axs[1].set_xlabel('Temperature (K)')
    axs[1].set_ylabel('$H / (RT)$')
    
    # S/R Plot
    axs[2].plot(T_all, sr_true, 'k-', linewidth=2)
    axs[2].plot(T_all, sr_fit, 'b--', linewidth=2)
    axs[2].axvline(Tmid, color='gray', linestyle=':')
    axs[2].set_title('$S / R$ Fit Verification')
    axs[2].set_xlabel('Temperature (K)')
    axs[2].set_ylabel('$S / R$')
    
    plt.tight_layout()
    plt.show()


def fmt_coef(x):
    # thermo.inp uses 16 char per column
    # sci notation doesn't use E but D
    return f"{x:16.9E}".replace("E", "D")


def formula_elements(name):

    core = name.rstrip("+-")

    order, counts = [], {}
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", core):
        if not sym:
            continue
        n = int(num) if num else 1
        if sym not in counts:
            order.append(sym)
        counts[sym] = counts.get(sym, 0) + n

    return [(sym, counts[sym]) for sym in order]


def molecular_weight(elements):
    mw = 0.0
    for sym, n in elements:
        mw += n * me if sym == "E" else ATOMIC_MASS[sym] * n
    return mw


def format_composition_line(n_intervals, elements, phase, mw, hf298, date_code="cccbdb"):
    elems = list(elements)[:5] + [("", 0.0)] * (5 - len(elements))
    elem_block = "".join(f"{sym:<2}{cnt:6.2f}" for sym, cnt in elems)
    return f"{n_intervals:2d} {date_code:<6} {elem_block} {phase:1d}{mw:13.7f}{hf298:15.3f}"


def format_temp_interval_line(t_low, t_high, h298_h0):
    exponents = " -2.0 -1.0  0.0  1.0  2.0  3.0  4.0  0.0"  # fixed powers of T, per NASA-9 spec
    return f"{t_low:11.3f}{t_high:11.3f}7{exponents}{h298_h0:17.3f}"


def format_coeff_lines(coeffs):
    a1, a2, a3, a4, a5, a6, a7, b1, b2 = coeffs
    line1 = "".join(fmt_coef(c) for c in (a1, a2, a3, a4, a5))
    line2 = fmt_coef(a6) + fmt_coef(a7) + " " * 16 + fmt_coef(b1) + fmt_coef(b2)
    return line1, line2


def format_thermo_entry(name, sp, comment, Tmin=200.0, Tmid=1000.0, Tmax=6000.0, newline="\n"):
    """
    Fit NASA-9 coefficients for species `sp` (as parsed by parse_data_file) and
    render them as a thermo.inp entry (name/comment line, composition line,
    then a 3-line block per temperature interval).
    """
    coeffs_low, coeffs_high = fit_nasa9(sp, Tmin, Tmid, Tmax)

    # Just used as reference value in thermo.inp
    _, h298_h0, _ = total_thermo(298.15, sp)

    elements = formula_elements(name)
    mw = molecular_weight(elements)
    hf298 = 0.0  # placeholder in meta :p

    lines = [
        f"{name:<18}{comment}",
        format_composition_line(2, elements, 0, mw, hf298),
        format_temp_interval_line(Tmin, Tmid, h298_h0),
        *format_coeff_lines(coeffs_low),
        format_temp_interval_line(Tmid, Tmax, h298_h0),
        *format_coeff_lines(coeffs_high),
    ]
    return newline.join(lines) + newline


def write_exotic_species(dat_path="./table/exotic_vib.dat", thermo_path=THERMO_PATH,
                          comment="Fit: partition.py NASA-9 from CCCBDB vib/rot data (B3LYP/aug-cc-pVDZ)."):
    """
    Fit every species in `dat_path` and insert their thermo.inp entries
    right before the "END PRODUCTS" marker in `thermo_path`.
    """
    species = parse_data_file(dat_path)

    # preserve the file's existing line-ending convention (thermo.inp ships as CRLF)
    with open(thermo_path, "rb") as f:
        newline = "\r\n" if b"\r\n" in f.read(4096) else "\n"

    with open(thermo_path, encoding="utf-8", errors="ignore", newline="") as f:
        lines = f.readlines()

    insert_at = next(i for i, ln in enumerate(lines) if ln.startswith("END PRODUCTS"))

    new_blocks = [format_thermo_entry(name, sp, comment, newline=newline) for name, sp in species.items()]
    lines[insert_at:insert_at] = new_blocks

    with open(thermo_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)

    return list(species.keys())


if __name__ == "__main__":
    #written = write_exotic_species()
    sp = parse_data_file("./table/exotic_vib.dat")
    plot_nasa9_fit(sp["ArH+"])
    #print(f"Wrote {len(written)} species to {THERMO_PATH}: {', '.join(written)}")