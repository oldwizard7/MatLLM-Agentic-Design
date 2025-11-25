#!/usr/bin/env python
"""Test evaluator directly without catching exceptions"""

import sys
from pathlib import Path
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.structure import CrystalStructure
from src.core.evaluator import StructureEvaluator

# Create structure
struct = CrystalStructure(
    formula='NaCl',
    lattice=np.eye(3) * 5.64,
    positions=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
    species=['Na', 'Cl']
)

# Create evaluator
config = {
    'evaluator': {
        'backend': 'orb',
        'model_name': 'orb-v3',
        'device': 'cpu',
        'relax': {
            'fmax': 0.1,
            'steps': 10,
            'optimizer': 'FIRE'
        }
    }
}

print("Creating evaluator...")
evaluator = StructureEvaluator(config)
print("✅ Evaluator created\n")

print("Relaxing structure...")
print(f"Initial is_valid: {struct.is_valid}")

# Temporarily bypass exception handling by calling the internal logic
from pymatgen.core import Structure
import ase
from ase.optimize import FIRE

try:
    # This is what relax_structure does
    pmg_struct = struct.to_pymatgen()
    print(f"✅ Converted to pymatgen")

    atoms = ase.Atoms(
        symbols=[str(s) for s in pmg_struct.species],
        positions=pmg_struct.cart_coords,
        cell=pmg_struct.lattice.matrix,
        pbc=True
    )
    print(f"✅ Converted to ASE")

    atoms.calc = evaluator.calculator
    print(f"✅ Calculator set")

    optimizer = FIRE(atoms, logfile=None)
    print(f"✅ Optimizer created")

    optimizer.run(fmax=0.1, steps=10)
    print(f"✅ Relaxation completed in {optimizer.get_number_of_steps()} steps")

    # Convert back
    relaxed_struct = Structure(
        lattice=atoms.cell[:],
        species=atoms.get_chemical_symbols(),
        coords=atoms.get_positions(),
        coords_are_cartesian=True
    )
    print(f"✅ Converted back to pymatgen")

    relaxed = CrystalStructure.from_pymatgen(relaxed_struct)
    print(f"✅ Converted to CrystalStructure")
    print(f"\n🎉 Complete pipeline works!")

except Exception as e:
    print(f"\n❌ Exception occurred: {e}")
    import traceback
    traceback.print_exc()
