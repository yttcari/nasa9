"""
Compare the nasa9 to kompot energy
Currently it prints and counts the missing reaction
Last update on 2026 Jul 10, there are 201/616 reactions are skipped :(

The commented out part is the plot for comparison graph
"""

import re
from kompot.nasa9 import get_H, get_coefficients
import matplotlib.pyplot as plt

T = 300
J_PER_EV_MOL = 96485.33212
H_CACHE = {}
MISSING_CACHE = {}


def parse_network(path="kompot/Network_new.txt"):
    reactions = {}
    energy_re = re.compile(r"#\s*(\d+)\s+energy\s*=\s*([-+0-9.eE]+)\s*eV")

    with open(path) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("@"):
                continue

            m = energy_re.match(line)
            if m:
                idx = int(m.group(1))
                if idx in reactions:
                    reactions[idx]["network_eV"] = float(m.group(2))
                continue

            if line.startswith("#"):
                continue

            cols = line.split(",", 10)
            if len(cols) < 10 or not cols[0].isdigit():
                continue

            idx = int(cols[0])
            reactions[idx] = {
                "reactants": [x for x in cols[1:4] if x],
                "products": [x for x in cols[4:8] if x],
                "network_eV": None,
            }

    return reactions


def reaction_enthalpy(reactants, products):
    h_reactants = sum(species_enthalpy(sp) for sp in reactants)
    h_products = sum(species_enthalpy(sp) for sp in products)
    return (h_products - h_reactants) / J_PER_EV_MOL


def species_enthalpy(sp):
    if sp in MISSING_CACHE:
        raise ValueError(MISSING_CACHE[sp])
    if sp not in H_CACHE:
        try:
            H_CACHE[sp] = get_H(sp, T)
        except Exception as e:
            MISSING_CACHE[sp] = str(e)
            raise
    return H_CACHE[sp]


def main():
    rows = []
    skipped = []
    zero_energy = []

    for idx, rxn in sorted(parse_network().items()):
        try:
            dH = reaction_enthalpy(rxn["reactants"], rxn["products"])
        except Exception as e:
            skipped.append((idx, str(e)))
            continue

        network = rxn["network_eV"]
        if network is None:
            skipped.append((idx, "no network energy comment"))
            continue

        if network == 0:
            zero_energy.append((idx, "Bruh 0 energy what reaction is this"))
            continue

        # network.txt stores approximately -dH
        rows.append({
            "idx": idx,
            "reaction": " + ".join(rxn["reactants"]) + " -> " + " + ".join(rxn["products"]),
            "dH_eV": -dH,
            "network_eV": network,
            "diff_eV": -dH - network,
        })

        if -dH - network > 10:
            print(f"LARGE DIFFERENCE: {idx}, {-dH}")

        if abs(rows[-1]['diff_eV']) > 5:
            for sp in rxn['reactants']:
                get_coefficients(species=sp, T=T, verbose=False)
            for sp in rxn['products']:
                get_coefficients(species=sp, T=T, verbose=False)

            print(
                f"{rows[-1]['idx']},"
                f"{rows[-1]['dH_eV']:.4f},"
                f"{rows[-1]['network_eV']:.4f},"
                f"{rows[-1]['diff_eV']:.4f},"
                f"{rows[-1]['reaction']}"
            )



    rows.sort(key=lambda x: abs(x["diff_eV"]), reverse=True)

    print(f"T = {T} K")
    print(f"compared: {len(rows)}")
    print(f"skipped: {len(skipped)}")
    print(f"Zero Energy: {len(zero_energy)}")
    print()
    print("idx,dH_eV,-dH_eV,network_eV,diff_eV,reaction")

    room_temp_E = []
    computed_E = []

    for row in rows:

        computed_E.append(row['dH_eV'])
        room_temp_E.append(row['network_eV'])
    
    # Plot the kompot energy against nasa9 computed energy
    plt.scatter(room_temp_E, computed_E)
    plt.xlabel("Reaction Energy in Kompot")
    plt.ylabel("Nasa-9 Coefficient")
    plt.plot(room_temp_E, room_temp_E, 'r--')
    plt.show()
    
    if skipped:
        print("\nskipped")
        for idx, reason in skipped:
            print(f"{idx}: {reason}")
            
    if MISSING_CACHE:
        output_path = "./table/missing_species.dat"
        with open(output_path, "w", encoding="utf-8") as f_out:
            for species in sorted(MISSING_CACHE.keys()):
                f_out.write(f"{species}\n")
    else:
        print("\nNo missing species found.")
    
if __name__ == "__main__":
    main()
