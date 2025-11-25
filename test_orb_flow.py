#!/usr/bin/env python
"""Test the exact evaluator flow"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.structure import CrystalStructure
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
import ase
from ase.optimize import FIRE
from pymatgen.core import Structure

print("Loading ORB model...")
orbff = pretrained.orb_v3_conservative_20_mpa(device='cpu')
calculator = ORBCalculator(orbff, device='cpu')
print("✅ Model loaded\n")

# Create CrystalStructure (like in the test)
print("Creating CrystalStructure...")
struct = CrystalStructure(
    formula='NaCl',
    lattice=np.eye(3) * 5.64,
    positions=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
    species=['Na', 'Cl']
)
print(f"Structure: {struct.formula}")
print(f"Lattice:\n{struct.lattice}")
print(f"Positions:\n{struct.positions}")
print(f"Species: {struct.species}\n")

# Convert to pymatgen (like evaluator does)
print("Converting to pymatgen...")
try:
    pmg_struct = struct.to_pymatgen()
    print(f"✅ Pymatgen structure created")
    print(f"   Formula: {pmg_struct.composition.reduced_formula}")
    print(f"   Species: {[str(s) for s in pmg_struct.species]}")
    print(f"   Cart coords:\n{pmg_struct.cart_coords}\n")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Convert to ASE (like evaluator does)
print("Converting to ASE...")
try:
    atoms = ase.Atoms(
        symbols=[str(s) for s in pmg_struct.species],
        positions=pmg_struct.cart_coords,
        cell=pmg_struct.lattice.matrix,
        pbc=True
    )
    print(f"✅ ASE atoms created")
    print(f"   Formula: {atoms.get_chemical_formula()}")
    print(f"   Positions:\n{atoms.get_positions()}\n")
except Exception as e:
    print(f"❌ ASE conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Set calculator
print("Setting calculator...")
atoms.calc = calculator
print("✅ Calculator set\n")

# Calculate energy
print("Calculating energy...")
try:
    energy = atoms.get_potential_energy()
    print(f"✅ Energy: {energy:.4f} eV\n")
except Exception as e:
    print(f"❌ Energy calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Run relaxation
print("Running relaxation...")
try:
    optimizer = FIRE(atoms, logfile=None)
    optimizer.run(fmax=0.1, steps=10)
    print(f"✅ Relaxation successful!")
    print(f"   Steps: {optimizer.get_number_of_steps()}")
    print(f"   Final energy: {atoms.get_potential_energy():.4f} eV\n")
except Exception as e:
    print(f"❌ Relaxation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Convert back to CrystalStructure
print("Converting back to CrystalStructure...")
try:
    relaxed_struct = Structure(
        lattice=atoms.cell[:],
        species=atoms.get_chemical_symbols(),
        coords=atoms.get_positions(),
        coords_are_cartesian=True
    )
    relaxed = CrystalStructure.from_pymatgen(relaxed_struct)
    print(f"✅ Conversion successful")
    print(f"   Formula: {relaxed.formula}")
    print(f"\n🎉 Full pipeline works!")
except Exception as e:
    print(f"❌ Back-conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
