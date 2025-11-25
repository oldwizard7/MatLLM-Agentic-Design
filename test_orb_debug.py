#!/usr/bin/env python
"""Debug ORB integration"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.structure import CrystalStructure
from src.core.evaluator import StructureEvaluator

# Create test structure (NaCl)
struct = CrystalStructure(
    formula='NaCl',
    lattice=np.eye(3) * 5.64,
    positions=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
    species=['Na', 'Cl']
)

# Create evaluator with ORB backend
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

print("Testing relaxation...")
try:
    relaxed = evaluator.relax_structure(struct)
    print(f"Relaxation complete. is_valid = {relaxed.is_valid}")

    if relaxed.is_valid:
        print("✅ Success!")
        energy = evaluator.calculate_energy(relaxed)
        print(f"Energy: {energy:.4f} eV/atom")
    else:
        print("❌ Failed - structure marked as invalid")

except Exception as e:
    print(f"❌ Exception during relaxation:")
    import traceback
    traceback.print_exc()
