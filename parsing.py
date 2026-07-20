import re

with open("./reaction_io/reaction.txt") as f:
    text = f.read()

token_pattern = re.compile(r'[A-Za-z][A-Za-z0-9\$\{\}\^_\\+\-\(\)]*')   

def normalize(tok):
    tok = tok.replace('$', '').replace('\\', '')
    tok = tok.strip()
    return tok

species = set()
for line in text.splitlines():
    parts = re.split(r'\\rightarrow|\\to', line)
    if len(parts) < 2:
        continue
    both_sides = re.split(r'&', line)
    eq = None
    for p in both_sides:
        if '\\rightarrow' in p:
            eq = p
            break
    if eq is None:
        continue
    for tok in re.split(r'(?<!\^)\+', eq.split('\\rightarrow')[0] + '+' + eq.split('\\rightarrow')[1]):
        tok = normalize(tok)
        tok = re.sub(r'^\s*M\s*$', '', tok)          
        if tok and tok != 'M':
            species.add(tok)

with open("./reaction_io/species.txt", "w") as f:
    for tok in species:
        tok = tok.replace("^", "").replace("_", "")
        f.write(tok + "\n")