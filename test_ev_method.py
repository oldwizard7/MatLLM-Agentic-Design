#!/usr/bin/env python
"""Test evaluator method directly"""

import sys
from pathlib import Path
import numpy as np
import logging

# Set up logging to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)

# Silence verbose libraries
logging.getLogger('filelock').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

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

print("="*60)
print("Creating evaluator...")
print("="*60)
evaluator = StructureEvaluator(config)

print("\n" + "="*60)
print("Calling relax_structure method...")
print("="*60)
relaxed = evaluator.relax_structure(struct)

print("\n" + "="*60)
print("Result:")
print("="*60)
print(f"is_valid: {relaxed.is_valid}")
print(f"is_relaxed: {relaxed.is_relaxed}")
print(f"formula: {relaxed.formula}")
print(f"metadata: {relaxed.metadata}")

if relaxed.is_valid:
    print("\n✅ SUCCESS!")
else:
    print("\n❌ FAILED - structure marked as invalid")
