import config as cfg
import numpy as np
import polars as pl
import regex as re
import pandas as pd
import csv
import nasa9
import Constants as c

def extract_species(ext):
    """
    Reads the species_densities file from kompot paths finalstate. Returns
    a dataframe with column names as index.
    """
    filename = cfg.kompot_path +"Projects/" + cfg.current_project  + "/" + ext + "/FinalState/species_densities.dat"

    with open(filename, "r") as f:
        content = [line.strip() for line in f.readlines()]

    header_line = content[2]
    column_names = []
    for word in header_line.split():
        if "_(" in word:
            column_names.append(word.split("_")[0])
        else:
            column_names.append(word)

    data_lines = content[3:]
    data_matrix = np.array([line.split() for line in data_lines])
    columns_names = [(name, pl.Float32) for name in column_names]
    df = pl.DataFrame(data=data_matrix, schema=columns_names)
    df = df.with_columns(
        pl.sum_horizontal([pl.col(c) for c in column_names[4:]]).alias("Number density")
    )
    return df

def extract_profile(ext):
    """Takes the thermalProperties and returns the properties in a dataframe."""

    filename = cfg.kompot_path +"Projects/" + cfg.current_project  + "/" + ext + "/FinalState/ThermalProperties.dat"
    print(filename)
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    line1 = content[0].split("=")
    nHeader = int(line1[1]) +2
    line2 = content[1].split("=")
    nrGrid = int(line2[1])
    
    columns_names = []
    # loop over every line in header
    for line in content[0:nHeader]:
        if ":" in line:
            line = line.split(":")[1].strip().split("(")[0].rstrip()
            columns_names.append(line)
    data_matrix = np.array([col.split() for col in content[nHeader : nrGrid + nHeader]])
    columns_names = [(name, pl.Float32) for name in columns_names]
    thermal_df = pl.DataFrame(data=data_matrix, schema=columns_names)

    return thermal_df


"""
Parser for KIDA/PHIDRATES-style chemical network text files (like Kompot's
`network` input file) into the pandas DataFrame shape expected by
make_kompot_network(df, ext).

Usage:
    from kida_network_parser import parse_kida_network
    df, variables = parse_kida_network("network")
    make_kompot_network(df, ext)
"""



def parse_kida_network(filepath, default_tmin=None, default_tmax=None):
    """
    Parse a KIDA/PHIDRATES-style reaction network file into a DataFrame with
    columns: index, type, reaction, alpha, beta, gamma, ref, tmin, tmax, energy
    (plus two extra informational columns, see below), ready to hand to
    make_kompot_network().

    Handles:
      - Standard thermal reactions: idx,r,r,r,p,p,p,p,tmin,tmax,rate
      - @photo_start / @photo_stop blocks (rate given as @xsecFile=<path>)
      - @photoelectron_start / @photoelectron_stop blocks
        (rate given as PhotoelectronRates(n))
      - @format: directives that redefine the column layout mid-file
      - @var: definitions (returned separately as `variables`, e.g.
        total_density, TgasNI, TgasNE, TgasIE -- these aren't reactions, just
        helper expressions used inside rate formulas)
      - '# <idx> energy = X eV' / '# <idx> bonding_energy = X eV' trailer
        comments
      - Free-text comment lines preceding a reaction, folded into `ref`

    Parameters
    ----------
    filepath : str
        Path to the network file.
    default_tmin, default_tmax : float, optional
        If given, substituted wherever the file says "NONE" for Tmin/Tmax.
        Left as None otherwise.

    Returns
    -------
    df : pandas.DataFrame
        Columns: index, type, reaction, alpha, beta, gamma, ref, tmin, tmax,
        energy, given_energy_eV, xsec_file.
    variables : dict
        The @var: definitions found in the file (name -> expression string).

    Assumptions worth double-checking against what make_kompot_network
    actually expects:

    1. Rate expressions in this file (e.g. "1.5e-14 * Tneut *
       exp(-3270.0/Tneut)") are arbitrary formulas, not a clean
       alpha*(T/300)^beta*exp(-gamma/T) triple. I put the *whole expression*
       into the `alpha` column as a string and leave `beta`/`gamma` as None.
       If make_kompot_network needs numeric alpha/beta/gamma separately,
       these expressions would need per-reaction fitting/parsing instead --
       let me know and I can add that.
    2. `energy` is always set to 0, matching your docstring ("Calculates
       energy itself"). The eV values already present in the file are parsed
       out separately into `given_energy_eV` so you can cross-check against
       whatever Kompot computes, rather than silently discarding them.
    3. `type` is inferred as one of: "normal", "three_body" (rate expression
       contains total_density), "photo" (inside @photo_start/stop, rate is an
       @xsecFile reference), "photoelectron" (inside @photoelectron_start/
       stop). Rename/remap these to whatever Kompot's `type` column actually
       expects if different.
    4. `ref` is a best-effort join of the free-text comment lines
       immediately preceding a reaction (source/change notes like "changed
       rate from Tian to KIDA"). It resets at "----" separator lines, so it's
       reasonably clean, but isn't guaranteed to always be exactly the
       intended reference -- worth a quick visual spot check.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    variables = {}
    rows = []

    # Default layout before any @format: directive appears (matches the
    # header comment "indx,r,r,r,p,p,p,p,minT,maxT,k").
    current_format = ["idx", "r", "r", "r", "p", "p", "p", "p", "tmin", "tmax", "rate"]
    section = "thermal"  # thermal | photo | photoelectron
    comment_buffer = []

    energy_re = re.compile(r"^#\s*(\d+)\s+(energy|bonding_energy)\s*=\s*([-+0-9.eE]+)\s*eV")

    def _num(v):
        if v is None:
            return None
        v = v.strip()
        if v.upper() == "NONE" or v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return v

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#"):
            m = energy_re.match(line)
            if m:
                idx = int(m.group(1))
                ev = float(m.group(3))
                if rows and rows[-1]["index"] == idx:
                    rows[-1]["given_energy_eV"] = ev
                continue
            if set(line) <= {"#", "-"}:
                comment_buffer = []
                continue
            comment_buffer.append(line.lstrip("#").strip())
            continue

        if line.startswith("@var:"):
            body = line.split(":", 1)[1].strip()
            if "=" in body:
                k, v = body.split("=", 1)
                variables[k.strip()] = v.strip()
            continue

        if line.startswith("@format:"):
            current_format = [c.strip() for c in line.split(":", 1)[1].split(",")]
            continue

        if line.startswith("@photo_start"):
            section = "photo"
            continue
        if line.startswith("@photo_stop"):
            section = "thermal"
            continue
        if line.startswith("@photoelectron_start"):
            section = "photoelectron"
            continue
        if line.startswith("@photoelectron_stop"):
            section = "thermal"
            continue

        if not re.match(r"^\d+,", line):
            # unrecognized directive/stray line - skip rather than crash
            continue

        fields = line.split(",", len(current_format) - 1)
        if len(fields) != len(current_format):
            continue
        # positional pairs (not a dict!) since column names like "r","r","r"
        # repeat and would collide as dict keys
        pairs = list(zip(current_format, fields))

        idx = int(pairs[0][1].strip())
        reactants, products = [], []
        tmin_raw = tmax_raw = None
        for col, val in pairs[1:-1]:
            val = val.strip()
            cl = col.lower()
            if cl in ("tmin", "mint"):
                tmin_raw = val
                continue
            if cl in ("tmax", "maxt"):
                tmax_raw = val
                continue
            if not val:
                continue
            if cl.startswith("r"):
                reactants.append(val)
            elif cl.startswith("p"):
                products.append(val)

        reaction_str = " + ".join(reactants) + " -> " + " + ".join(products)
        rate_expr = pairs[-1][1].strip()

        xsec_file = None
        if rate_expr.startswith("@xsecFile="):
            xsec_file = rate_expr.split("=", 1)[1].strip()

        rtype = {"photo": "photo", "photoelectron": "photoelectron"}.get(section)
        if rtype is None:
            rtype = "three_body" if "total_density" in rate_expr else "normal"

        ref = " / ".join(comment_buffer) if comment_buffer else None
        comment_buffer = []

        tmin = _num(tmin_raw)
        tmax = _num(tmax_raw)
        if tmin is None and default_tmin is not None:
            tmin = default_tmin
        if tmax is None and default_tmax is not None:
            tmax = default_tmax

        rows.append({
            "index": idx,
            "type": rtype,
            "reaction": reaction_str,
            "alpha": rate_expr,
            "beta": None,
            "gamma": None,
            "ref": ref,
            "tmin": tmin,
            "tmax": tmax,
            "energy": 0,
            "given_energy_eV": None,
            "xsec_file": xsec_file,
        })

    df = pd.DataFrame(rows, columns=[
        "index", "type", "reaction", "alpha", "beta", "gamma",
        "ref", "tmin", "tmax", "energy", "given_energy_eV", "xsec_file"
    ])
    return df, variables

def make_kompot_network(df, ext):
   """
   Makes a kompot chemical network from a dataframe with index, reaction, alpha
   , beta, gamma basically. Calculates energy itself.
   """
   
   df_e = calc_energies(df, T=300)
   
   # Lists to hold our photo stuff to process at the very end
   ph_list = []
   pe_list = []
   
   # Determine the highest index among regular reactions to safely start dynamic numbering
   regular_df = df_e[~df_e["type"].isin(['ph', 'pe'])]
   max_reg_idx = int(regular_df["index"].max()) if not regular_df.empty else 0
   running_idx = max_reg_idx + 1
   
   with open("Network"+ ext, "w") as f:
        f.write("""@var: total_density = sum(n(:))
@var: TgasNI = 0.5 * ( Tneut + Tion )
@var: TgasNE = 0.5 * ( Tneut + Telec )
@var: TgasIE = 0.5 * ( Tion + Telec )
# T in the rate coefficient equation depends on the type of reactant
# e.g., if a neutral and an ion is on the LHS of the equation use TgasNI
#------------------------------------------------------------------------------------------
# changed rates from Tian values to Fox & Sung values
# indx,r,r,r,p,p,p,p,minT,maxT,k
# energy of reaction
# indx,r,r,r,p,p,p,p,minT,maxT,k
# energy of reaction
                """)

        for _, row in df_e.iterrows():
            idx = int(row["index"]) # Keep original forward numbering for regular reactions
            reaction = row["reaction"]
            alpha = float(row["alpha"])
            beta = float(row["beta"])
            gamma = -float(row["gamma"])
            r_type = row["type"]
            t_min = row["tmin"]
            t_max = row["tmax"]
            energy = -row["energy"]
            
            # Separate photo reactions to be formatted and numbered later
            if r_type in ['ph', 'pe']:
                if r_type == 'ph':
                    ph_list.append(row)
                else:
                    pe_list.append(row)
                continue

            # Format reaction       
            reactants_str, products_str = reaction.split("=>")
            reactants = [s.strip() for s in reactants_str.split(" + ") if s.strip()]
            products = [s.strip() for s in products_str.split(" + ") if s.strip()]

            # Pad to 3 reactants/products for regular equations
            reactants += [""] * (3 - len(reactants))
            products += [""] * (4 - len(products))
            # Just put rb here to not get rr right now :)
            if ("rb" in r_type) and (energy != 0):
                # Format rate expression
                if beta == 0 and gamma == 0:
                    rate_expr = f"{alpha:.1e}"
                elif gamma == 0:
                    rate_expr = f"{alpha:.1e} * (Tneut/300)**({beta})"
                elif beta == 0:
                    rate_expr = f"{alpha:.1e} * exp({gamma:.1f} / Tneut)"
                else:
                    rate_expr = f"{alpha:.1e} * (Tneut/300)**({beta}) * exp({gamma:.1f} / Tneut)"
    
                # Write block
                f.write("#" + "-" * 90 + "\n")
                r = reactants[:3]
                p = products[:4]
                # Write the full line
                f.write(f"{idx},{r[0]},{r[1]},{r[2]},{p[0]},{p[1]},{p[2]},{p[3]},{t_min},{t_max},{rate_expr}\n")
                f.write(f"# {idx} energy = {energy} eV\n\n")
                
                # Write reverse reaction using dynamically generated running_idx
                rate_expr_rev = f'({rate_expr})/exp({-energy*c.KJMOL_TO_EV}/({c.R}*Tneut))'
                f.write("#" + "-" * 37 + "Reverse Reaction" + "-" * 37 + "\n")
                f.write(f"{running_idx},{r[0]},{r[1]},{r[2]},{p[0]},{p[1]},{p[2]},{p[3]},{t_min},{t_max},{rate_expr_rev}\n")
                f.write(f"# {running_idx} energy = {-energy} eV\n\n")
                
                # If reverse reactions are enabled in the future, we advance the running index here
                running_idx += 1
            else:    
                # Format rate expression
                if beta == 0 and gamma == 0:
                    rate_expr = f"{alpha:.1e}"
                elif gamma == 0:
                    rate_expr = f"{alpha:.1e} * (Tneut/300)**({beta})"
                elif beta == 0:
                    rate_expr = f"{alpha:.1e} * exp({gamma:.1f} / Tneut)"
                else:
                    rate_expr = f"{alpha:.1e} * (Tneut/300)**({beta}) * exp({gamma:.1f} / Tneut)"

                # Write block
                f.write("#" + "-" * 90 + "\n")
               
                r = reactants[:3]
                p = products[:4]
                # Write the full line
                f.write(f"{idx},{r[0]},{r[1]},{r[2]},{p[0]},{p[1]},{p[2]},{p[3]},{t_min},{t_max},{rate_expr}\n")
                f.write(f"# {idx} energy = {energy} eV\n\n")

        # Now that the standard loop is over, build and assign continuous numbers to the photo reactions
        pe_block = ""
        for row in pe_list:
            reaction = row["reaction"]
            reactants_str, products_str = reaction.split("=>")
            reactants = [s.strip() for s in reactants_str.split(" + ") if s.strip()]
            products = [s.strip() for s in products_str.split(" + ") if s.strip()]
            
            r = reactants[:1] + [''] * (1 - len(reactants))
            p = products[:3] + [''] * (3 - len(products))
            
            t_min = row["tmin"]
            t_max = row["tmax"]
            rate_expr = str(row["ref"])
            energy = row["energy"]
            
            block_str = f"{running_idx},{r[0]},{p[0]},{p[1]},{p[2]},NONE,NONE,{rate_expr}\n"
            block_str += f"# {running_idx} energy = {0} eV\n\n"
            pe_block += block_str
            running_idx += 1  # Increment for the next photo reaction

        ph_block = ""
        for row in ph_list:
            reaction = row["reaction"]
            reactants_str, products_str = reaction.split("=>")
            reactants = [s.strip() for s in reactants_str.split(" + ") if s.strip()]
            products = [s.strip() for s in products_str.split(" + ") if s.strip()]
            
            r = reactants[:1] + [''] * (1 - len(reactants))
            p = products[:3] + [''] * (3 - len(products))
            
            t_min = row["tmin"]
            t_max = row["tmax"]
            rate_expr = str(row["ref"])
            energy = row["energy"]
            
            block_str = f"{running_idx},{r[0]},{p[0]},{p[1]},{p[2]},NONE,NONE,{rate_expr}\n"
            block_str += f"# {running_idx} bonding_energy = {energy} eV\n\n"
            ph_block += block_str
            running_idx += 1  # Increment for the next photo reaction
            
        if pe_block:
            f.write("#" + "-" * 90 + "\n")
            f.write("@photoelectron_start\n")
            f.write("@format:idx,R,P,P,P,Tmin,Tmax,rate\n\n")
            f.write(pe_block)
            print(pe_block)
            f.write("@photoelectron_stop\n\n")
            
        if ph_block:
            f.write("#" + "-" * 90 + "\n")
            f.write("@photo_start\n")
            f.write("@format:idx,R,P,P,P,Tmin,Tmax,rate\n\n")
            f.write(ph_block)
            f.write("@photo_stop\n\n")
            
   return True

def find_current_species(filepath):
    """
    Checks what species are in your kompot. 
    """
    species_names = []
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Check if line starts with the REGEX (Guns blazing in background)
            if line.startswith("if (namesSpecies(iSpecies) =="):
                # Use regex (Guns blazing in background) to extract molecule name
                match = re.search(r"==\s*'([^']+)'", line)
                if match:
                    species_names.append(match.group(1))
    
    return species_names

def is_single_atom(species):
    """Returns True if the species is just a single atom (e.g., 'H', 'O', 'HE')"""
    flat = flatten_formula(species)
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', flat)
    return len(tokens) == 1 and (tokens[0][1] == '' or tokens[0][1] == '1')


def latexify(species):
    """Converts species string to LaTeX style, e.g. H2O -> H$_2$O"""
    species = species.strip()

    if(species == 'HE'):
        return 'He'
    # REG (say it with me) EX
    species = re.sub(r'\(([^\)]*)\)', '', species)

    # Extract and preserve charge suffix
    charge = ''
    if species[-1] in '+-':
        charge = '^' + species[-1]
        species = species[:-1]
    elif species[-2:] in ['2+', '3+', '2-', '3-']:
        charge = '^' + species[-2:]
        species = species[:-2]

    parts = re.findall(r'([A-Z][a-z]?)(\d*)', species)
    latex = ""
    for el, count in parts:
        if count:
            latex += f"{el}$_{count}$"
        else:
            latex += el
    if charge:
        latex += f"${charge}$"
    return latex


def extract_species_from_reactions(df):
    species_set = set()
    for reaction in df["reaction"]:
        # Remove whitespace and split by '+' or '=>' or '=' or '<=>' (in case other formats exist)
        parts = re.split(r'\+|=>|=|<=>', reaction)
        for part in parts:
            species = part.strip()
            if species:
                species_set.add(species)
    return species_set

def calc_mol_m(mol):
    """
    Reads a molecule as a string (flattened so (CH3)2 = C2H6) and returns the calculated molecular mass.
    """

    # Dictionary of atomic masses duh
    elements_dict = {
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011,
        'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
        'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.066, 'Cl': 35.453, 'Ar': 39.948,
        'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
        'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
        'Ga': 69.723, 'Ge': 72.631, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 84.798,
        'Rb': 84.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95,
        'Tc': 98.907, 'Ru': 101.07, 'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.414,
        'In': 114.818, 'Sn': 118.711, 'Sb': 121.760, 'Te': 126.7, 'I': 126.904, 'Xe': 131.294,
        'Cs': 132.905, 'Ba': 137.328, 'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.243,
        'Pm': 144.913, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.925, 'Dy': 162.500,
        'Ho': 164.930, 'Er': 167.259, 'Tm': 168.934, 'Yb': 173.055, 'Lu': 174.967, 'Hf': 178.49,
        'Ta': 180.948, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.085,
        'Au': 196.967, 'Hg': 200.592, 'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.980, 'Po': 208.982,
        'At': 209.987, 'Rn': 222.081, 'Fr': 223.020, 'Ra': 226.025, 'Ac': 227.028, 'Th': 232.038,
        'Pa': 231.036, 'U': 238.029, 'Np': 237, 'Pu': 244, 'Am': 243, 'Cm': 247, 'Bk': 247,
        'Ct': 251, 'Es': 252, 'Fm': 257, 'Md': 258, 'No': 259, 'Lr': 262, 'Rf': 261, 'Db': 262,
        'Sg': 266, 'Bh': 264, 'Hs': 269, 'Mt': 268, 'Ds': 271, 'Rg': 272, 'Cn': 285, 'Nh': 284,
        'Fl': 289, 'Mc': 288, 'Lv': 292, 'Ts': 294, 'Og': 294
    }

    # Use REGEX baby (Guns blazing in background)
    mol = flatten_formula(mol)
    pattern = r'([A-Z][a-z]?)(\d*)'
    tokens = re.findall(pattern, mol)
    mass = 0.0
    for element, count in tokens:
        if element not in elements_dict:
            raise ValueError(f"Unknown element: {element}")
        count = int(count) if count else 1
        mass += elements_dict[element] * count

    return mass

def read_enthalpies():
    

    # Read the CSV into a dictionary
    result = {}

    with open('../table/formation_H.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header (Species, Enthalpy (kJ/mol))
        
        for row in reader:
            if not row:  # Skip empty lines
                continue
            
            key = row[0].strip()  # Species name (e.g., "1CH2")
            
            # If there's a value, use it; otherwise, set to None
            if len(row) >= 2 and row[1].strip():
                try:
                    value = float(row[1].strip())  # Convert to float if possible
                except ValueError:
                    value = None  # Handle non-numeric values if needed
            else:
                value = None  # Missing value
            
            result[key] = value

    return result

def calc_energies(df, T=298.15):

    enthalpies = read_enthalpies()
    species_existing = enthalpies.keys()

    for i, row in df.iterrows():
        reaction_right, reaction_left = re.split(r'=>|<=>', row["reaction"])
        reactants_r = re.split(r'(?<= )\+(?= )', reaction_right)
        reactants_l = re.split(r'(?<= )\+(?= )', reaction_left)
        reactants_r = [r.strip() for r in reactants_r]
        reactants_l = [r.strip() for r in reactants_l]
        energies_l = []
        energies_r = []

        for r in reactants_l:
            try:
                H = nasa9.get_H(r, T) / 1000.0  # J/mol to kJ/mol
                energies_l.append(H)
            except Exception: # fallback
                print(f"{r} is not found in the NASA9 database. Fallback to default energy.")
                if r in species_existing and enthalpies[r] is not None:
                    energies_l.append(enthalpies[r])
                else:
                    energies_l.append('X')

        for r in reactants_r:
            try:
                H = nasa9.get_H(r, T) / 1000.0  # J/mol -> kJ/mol
                energies_r.append(H)
            except Exception:
                print(f"{r} is not found in the NASA9 database. Fallback to default energy.")
                if r in species_existing and enthalpies[r] is not None:
                    energies_r.append(enthalpies[r])
                else:
                    energies_r.append('X')

        if not (('X' in energies_l) or ('X' in energies_r)):
            df.at[i, "energy"] = (sum(energies_l) - sum(energies_r)) / c.KJMOL_TO_EV
        else:
            df.at[i, "energy"] = 0

    return df

def search_and_replace(file_path, search_word, replace_word):  
    try:  
        with open(file_path, 'r') as file:  
            file_contents = file.read()  

        updated_contents = file_contents.replace(search_word, replace_word)  

        with open(file_path, 'w') as file:  
            file.write(updated_contents)  
        print("Text replacement completed successfully.")  
    except FileNotFoundError:  
        print(f"Error: The file '{file_path}' was not found.")  
    except Exception as e:  
        print(f"An error occurred: {e}")  

def flatten_formula(mol):
    """
    Flattens a chemical formula:
    - Expands groupings like (CH3)2 → CH3CH3
    - Removes annotations like (1S), (aDelta)
    - Removes ionic charges like +, -, 2+
    """

    mol = mol.strip()
    mol = re.sub(r'^(cyc\-|oxy\-|tert\-|iso\-|neo\-)', '', mol, flags=re.IGNORECASE)
    # REGEX lesgo (Background rock music)
    while True:
        match = re.search(r'\(([^()]+)\)(\d+)', mol)
        if not match:
            break
        group, count = match.groups()
        mol = mol[:match.start()] + group * int(count) + mol[match.end():]

    mol = re.sub(r'\([^()]*\)', '', mol)
    mol = re.sub(r'[\+\-]\d*$', '', mol)

    return mol
           