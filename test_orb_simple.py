#!/usr/bin/env python
"""Simplest possible ORB test"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test just the ORB calculator directly
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
import ase
import numpy as np

print("Loading ORB model...")
orbff = pretrained.orb_v3_conservative_20_mpa(device='cpu')
calculator = ORBCalculator(orbff, device='cpu')
print("✅ Model loaded")

print("\nCreating NaCl structure...")
atoms = ase.Atoms(
    symbols=['Na', 'Cl'],
    positions=np.array([[0, 0, 0], [2.82, 2.82, 2.82]]),
    cell=np.eye(3) * 5.64,
    pbc=True
)
print(f"Structure: {atoms.get_chemical_formula()}")

print("\nSetting calculator...")
atoms.calc = calculator

print("\nCalculating energy...")
try:
    energy = atoms.get_potential_energy()
    print(f"✅ Energy: {energy:.4f} eV")
except Exception as e:
    print(f"❌ Energy calculation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nRunning relaxation...")
from ase.optimize import FIRE

try:
    optimizer = FIRE(atoms, logfile=None)
    optimizer.run(fmax=0.1, steps=10)
    final_energy = atoms.get_potential_energy()
    print(f"✅ Relaxation successful!")
    print(f"   Steps: {optimizer.get_number_of_steps()}")
    print(f"   Final energy: {final_energy:.4f} eV")
except Exception as e:
    print(f"❌ Relaxation failed: {e}")
    import traceback
    traceback.print_exc()
