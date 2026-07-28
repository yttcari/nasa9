import Helpers_new as h
import regex as re
import pandas as pd
import polars as pl
filepath = 'network.txt'


def read_kompot_network(file_path):
    """
    Reads a kompot chemical network file and parses it into a pandas DataFrame.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    records = []
    current_type = 'rr'  # Default to regular reaction
    
    # ---------------------------------------------------------
    # Pre-compile Regex patterns to evaluate the rate expressions
    # ---------------------------------------------------------
    # Matches exp(-gamma / T) or exp(gamma / T)
    gamma_re = re.compile(r"exp\(\s*(-?\d+\.?\d*)\s*/\s*T[a-zA-Z]*\s*\)")
    
    # Matches (T/300)**beta or (300/T)**beta
    beta_re_std = re.compile(r"\(\s*T[a-zA-Z]*/300\.?0?\s*\)\s*\*\*\s*\(?(-?\d+\.?\d*)\)?")
    beta_re_inv = re.compile(r"\(\s*300\.?0?\s*/\s*T[a-zA-Z]*\s*\)\s*\*\*\s*\(?(-?\d+\.?\d*)\)?")
    
    # Matches raw T**beta without the 300 denominator
    beta_re_raw = re.compile(r"(?<!/)\bT[a-zA-Z]*\b\s*\*\*\s*\(?(-?\d+\.?\d*)\)?")
    
    # Matches the leading numerical coefficient for Alpha
    alpha_re = re.compile(r"^[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

    for i, line in enumerate(lines):
        line = line.strip()
        
        # Track the active block type
        if line.startswith('@photo_start'):
            current_type = 'ph'
            continue
        elif line.startswith('@photoelectron_start'):
            current_type = 'pe'
            continue
        elif line.startswith('@photo_stop') or line.startswith('@photoelectron_stop'):
            current_type = 'rr'
            continue
            
        # Identify reaction rows (starting with an index number followed by a comma)
        if re.match(r"^\d+,", line):
            parts = [p.strip() for p in line.split(',')]
            is_photo = (current_type in ['ph', 'pe'])
            
            # Photo reactions have 8 columns (idx,R,P,P,P,Tmin,Tmax,rate)
            if is_photo and len(parts) >= 8:
                idx = int(parts[0])
                reactants = [r for r in parts[1:2] if r]
                products = [p for p in parts[2:5] if p]
                reaction = " + ".join(reactants) + " => " + " + ".join(products)
                tmin = 10.0 if parts[5] == 'NONE' else float(parts[5])
                tmax = 30000.0 if parts[6] == 'NONE' else float(parts[6])
                rate_expr = ",".join(parts[7:])
                # Initialize variables to 0[cite: 1]
                alpha, beta, gamma = 0.0, 0.0, 0.0
                
            # Regular reactions have 11 columns
            elif not is_photo and len(parts) >= 11:
                idx = int(parts[0])
                reactants = [r for r in parts[1:4] if r]
                products = [p for p in parts[4:8] if p]
                reaction = " + ".join(reactants) + " => " + " + ".join(products)
                tmin = 10.0 if parts[8] == 'NONE' else float(parts[8])
                tmax = 30000.0 if parts[9] == 'NONE' else float(parts[9])
                rate_expr = ",".join(parts[10:])
                # Initialize variables to 0[cite: 1]
                alpha, beta, gamma = 0.0, 0.0, 0.0
            else:
                continue
                
            # Parse rate expressions for regular reactions
            if current_type == 'rr' and not rate_expr.startswith('@') and 'PhotoelectronRates' not in rate_expr:
                
                # 1. Parse Gamma
                g_match = gamma_re.search(rate_expr)
                if g_match:
                    gamma = -float(g_match.group(1))
                    
                # 2. Parse Beta
                b_match_std = beta_re_std.search(rate_expr)
                b_match_inv = beta_re_inv.search(rate_expr)
                b_match_raw = beta_re_raw.search(rate_expr)
                
                if b_match_std:
                    beta = float(b_match_std.group(1))
                elif b_match_inv:
                    beta = -float(b_match_inv.group(1))  # Invert sign if it's (300/T)
                elif b_match_raw:
                    beta = float(b_match_raw.group(1))
                elif re.search(r"(?<!\w)T[a-zA-Z]*(?!\w)", rate_expr):
                    # Implicit linear dependence (e.g., ... * Tneut * ...)
                    beta = 1.0
                    
                # 3. Parse Alpha
                a_match = alpha_re.search(rate_expr)
                if a_match:
                    alpha = float(a_match.group())
                    if b_match_raw or (beta == 1.0 and not b_match_std and not b_match_inv):
                        alpha *= (300.0 ** beta)
            
            # Extract energy from the succeeding comment line
            energy = 0.0
            if i + 1 < len(lines):
                next_line = lines[i+1].strip()
                # Example target: # 1 energy = 1.40 eV OR # 133 bonding_energy = 9.79 eV
                e_match = re.search(rf"#\s*{idx}\s+.*energy\s*=\s*([-\d.]+)\s*eV", next_line)
                if e_match:
                    energy = float(e_match.group(1))
                    
            records.append({
                "index": idx,
                "type": current_type,
                "reaction": reaction,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "ref": rate_expr if is_photo else "", # Dump photo expressions into ref
                "tmin": tmin,
                "tmax": tmax,
                "energy": energy
            })
                
    return pd.DataFrame(records)





df = read_kompot_network('network.txt')
for i, row in df.iterrows():
    print(row['reaction'])
h.make_kompot_network(df, '_new.txt')
