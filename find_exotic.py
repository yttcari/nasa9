"""
Find compound in kompot/network.txt that 
are not in the nasa9 thermo.inp

Missing compound:
C4H, C4H3, ClCO3, H2C3N, H2D+, HC2N2, HCNS, HCS, HNCS, HS2, HSNO, HSO, HSO2, HSO

Last update: 2026 Jul 10
"""
from kompot.nasa9 import get_coefficients

def network_species(path="kompot/network.txt"):
    species = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue

            cols = line.split(",", 10)
            if len(cols) < 10 or not cols[0].strip().isdigit():
                continue

            for name in cols[1:8]:
                name = name.strip()
                if name:
                    species.add(name)
    return species


def species_type(name):
    if name == "e":
        return "electron"
    if "_" in name:
        return "electronic/excited"
    return "molecular/ground"


def unreadable_species(T=1000.0):
    missing = {"electron": [], "electronic/excited": [], "molecular/ground": []}

    for name in sorted(network_species()):
        try:
            get_coefficients(name, T)
        except Exception:
            missing[species_type(name)].append(name)

    return missing


if __name__ == "__main__":
    missing = unreadable_species()

    for kind, names in missing.items():
        print(f"\n{kind} ({len(names)})")
        print(", ".join(names) if names else "none")

    print(f"\ntotal unreadable: {sum(len(v) for v in missing.values())}")
