# Molecular templates

PACKMOL needs one PDB per molecular species. Two kinds live here:

1. **Pre-supplied:** `Li.pdb` (single atom, hand-written).
2. **Generate from SMILES:** use `scripts/build_molecule.py`. Examples
   below produce templates for the rest of the LiPF6 / LiTFSI / DME / PDOL /
   DAC / PEI / BPEI inventory.

## SMILES recipes

```bash
# Anions and small molecules
python scripts/build_molecule.py --smiles "F[P-](F)(F)(F)(F)F"           --out templates/molecules/PF6.pdb        --name PF6
python scripts/build_molecule.py --smiles "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F" \
                                 --out templates/molecules/TFSI.pdb       --name TFS
python scripts/build_molecule.py --smiles "COCCOC"                       --out templates/molecules/DME.pdb        --name DME

# PDOL = poly(1,3-dioxolane); a 5-mer is a usable polymer template.
# (Open the bare 1,3-dioxolane ring and chain 5 units; SMILES below is a
# linear 5-mer terminated with -OH on each end.)
python scripts/build_molecule.py --smiles "OCCOCOCCOCOCCOCOCCOCOCCO"     --out templates/molecules/PDOL_5mer.pdb  --name PDL

# Small DAC monomer placeholder — dialdehyde glucose unit. Real DAC
# scaffolds are oxidised cellulose chains; for a v0.6 PoC, use a single
# dialdehyde-glucose ring and oligomerise externally if needed.
python scripts/build_molecule.py --smiles "O=CC(O)C(O)C(O)C(O)C=O"       --out templates/molecules/DAC_unit.pdb   --name DAC

# CgPEI / CgBPEI: linear / branched PEI repeat units. Replace these with
# actual graft chemistry once you pin it down.
python scripts/build_molecule.py --smiles "NCCNCCNCCNCCN"                --out templates/molecules/PEI_4mer.pdb   --name PEI
python scripts/build_molecule.py --smiles "NCCN(CCN)CCN(CCN)CCN"         --out templates/molecules/BPEI_branch.pdb --name BPI
```

The `--name` argument sets a 3-letter residue name in the PDB so PACKMOL
can label molecules cleanly.

## Editing existing PDBs

PACKMOL accepts any well-formed PDB. If you have monomers from a literature
source or built in Avogadro / RDKit notebooks, drop them here and reference
them by base name (without `.pdb`) from the `composition` section of your
system YAML.
