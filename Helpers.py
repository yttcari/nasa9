import config as cfg
import numpy as np
import polars as pl
import regex as re
import csv
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

def make_kompot_network(df, ext):
   """
   Makes a kompot chemical network from a dataframe with index, reaction, alpha
   , beta, gamma basically. Calculates energy itself. Dataframe (pandas) should have the structure:
               data({
                "index": index,
                "type": reaction_type,
                "reaction": reaction,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "ref": ref,
                "tmin": 10,
                "tmax": 30000,
                "energy": 0
            })
   
   """
   
   df_e = calc_energies(df)
   with open("TestData/Models/Networks/network_"+ ext, "w") as f:
        index = 1
        for _, row in df_e.iterrows():
            reaction = row["reaction"]
            alpha = row["alpha"]
            beta = row["beta"]
            gamma = row["gamma"]
            r_type = row["type"]
            t_min = row["tmin"]
            t_max = row["tmax"]
            energy = row["energy"]
            # Format reaction   
            reactants_str, products_str = reaction.split("=>")
            reactants = [s.strip() for s in reactants_str.split("+") if s.strip()]
            products = [s.strip() for s in products_str.split("+") if s.strip()]

            # Pad to 3 reactants/products
            reactants += [""] * (3 - len(reactants))
            products += [""] * (3 - len(products))
            if  ("rr" in r_type) and (energy != 0):
                # Format rate expression
                if beta == 0 and gamma == 0:
                    rate_expr = f"{alpha:.1e}"
                elif gamma == 0:
                    rate_expr = f"{alpha:.1e} * (Tneut/300)**({beta})"
                elif beta == 0:
                    rate_expr = f"{alpha:.1e} * exp(-{gamma:.1f} / Tneut)"
                else:
                    rate_expr = f"{alpha:.1e} * (Tneut/300)**({beta}) * exp(-{gamma:.1f} / Tneut)"
    
                # Compute energy
                #energy = 0
    
                # Write block
                f.write("#" + "-" * 90 + "\n")
                r = reactants[:3] + [''] * (3 - len(reactants))
                p = products[:4] + [''] * (4 - len(products))
                # Write the full line
                f.write(f"{index},{r[0]},{r[1]},{r[2]},{p[0]},{p[1]},{p[2]},{p[3]},{t_min},{t_max},{rate_expr}\n")
                f.write(f"# {index} energy = {energy} eV\n\n")
                index += 1
                
                # Write reverse reaction
                rate_expr_rev = f'({rate_expr})/exp({-energy*c.KJMOL_TO_EV}/({c.R}*Tneut))'
                                # Write block
                f.write("#" + "-" * 37 + "Reverse Reaction" + "-" * 37 + "\n")
                
                r = reactants[:3] + [''] * (3 - len(reactants))
                p = products[:4] + [''] * (4 - len(products))
                # Write the full line
                f.write(f"{index},{r[0]},{r[1]},{r[2]},{p[0]},{p[1]},{p[2]},{p[3]},{t_min},{t_max},{rate_expr}\n")
                f.write(f"# {index} energy = {-energy} eV\n\n")
                index += 1
            else:    
                # Format rate expression
                if beta == 0 and gamma == 0:
                    rate_expr = f"{alpha:.1e}"
                elif gamma == 0:
                    rate_expr = f"{alpha:.1e} * (Tneut/300)**({beta})"
                elif beta == 0:
                    rate_expr = f"{alpha:.1e} * exp(-{gamma:.1f} / Tneut)"
                else:
                    rate_expr = f"{alpha:.1e} * (Tneut/300)**({beta}) * exp(-{gamma:.1f} / Tneut)"

                # Compute energy
                #energy = 0

                # Write block
                f.write("#" + "-" * 90 + "\n")
               
                r = reactants[:3] + [''] * (3 - len(reactants))
                p = products[:4] + [''] * (4 - len(products))
                # Write the full line
                f.write(f"{index},{r[0]},{r[1]},{r[2]},{p[0]},{p[1]},{p[2]},{p[3]},{t_min},{t_max},{rate_expr}\n")
                f.write(f"# {index} energy = {energy} eV\n\n")
                index += 1
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

    with open('/home/nils-martin/Repos/auxfiles/TestData/species_enthalpies.csv', 'r') as f:
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

def calc_energies(df):

    enthalpies = read_enthalpies()
    species_existing = enthalpies.keys()
    energies = []
    for i, row in df.iterrows():
        reaction_right, reaction_left = re.split(r'=>|<=>', row["reaction"])
        reactants_r = re.split(r'(?<= )\+(?= )', reaction_right)
        reactants_l = re.split(r'(?<= )\+(?= )', reaction_left)
        reactants_r = [r.strip() for r in reactants_r]
        reactants_l = [r.strip() for r in reactants_l]
        energies_r = []
        energies_l = []
        for r in reactants_l:
            if r in species_existing:
                energies_r.append(enthalpies[r])
            else:
                energies_r.append('X')
        for r in reactants_r:
            if r in species_existing:
                energies_l.append(enthalpies[r])
            else:
                energies_l.append('X')
        if not(('X' in energies_l) | ('X' in energies_r)):
            df.at[i,"energy"] = (sum(energies_l) - sum(energies_r))/c.KJMOL_TO_EV
        else:
            df.at[i,"energy"] = 0
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
           