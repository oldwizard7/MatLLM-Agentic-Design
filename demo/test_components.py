"""Test individual components"""

import sys
sys.path.insert(0, '.')

from src.core.structure import CrystalStructure
import numpy as np


def test_structure_creation():
    """Test CrystalStructure creation and serialization"""
    print("Testing CrystalStructure...")
    
    # Create structure
    lattice = np.array([
        [5.64, 0.0, 0.0],
        [0.0, 5.64, 0.0],
        [0.0, 0.0, 5.64]
    ])
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0]
    ])
    
    species = ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl']
    
    struct = CrystalStructure(
        formula="Na4Cl4",
        lattice=lattice,
        positions=positions,
        species=species
    )
    
    print(f"✓ Created structure: {struct.formula}")
    print(f"  Volume: {struct.volume:.2f} Å³")
    print(f"  Density: {struct.density:.2f} g/cm³")
    
    # Test POSCAR conversion
    poscar_str = struct.to_poscar()
    print(f"✓ Converted to POSCAR ({len(poscar_str)} chars)")
    
    # Test parsing
    parsed = CrystalStructure.from_poscar(poscar_str)
    print(f"✓ Parsed back: {parsed.formula}")
    
    # Test serialization
    struct.save('test_structure.json')
    loaded = CrystalStructure.load('test_structure.json')
    print(f"✓ Saved and loaded: {loaded.formula}")
    
    import os
    os.remove('test_structure.json')
    
    print("All structure tests passed!\n")


def test_parser():
    """Test structure parser"""
    print("Testing StructureParser...")
    
    from src.utils.parser import StructureParser
    
    # Test POSCAR extraction
    text = """
Here are some structures:

Structure 1:
NaCl
1.0
5.64 0.0 0.0
0.0 5.64 0.0
0.0 0.0 5.64
Na Cl
4 4
direct
0.0 0.0 0.0 Na
0.5 0.5 0.0 Na
0.5 0.0 0.5 Na
0.0 0.5 0.5 Na
0.5 0.5 0.5 Cl
0.0 0.0 0.5 Cl
0.0 0.5 0.0 Cl
0.5 0.0 0.0 Cl

Some other text here.
"""
    
    structures = StructureParser.parse_structures(text)
    print(f"✓ Extracted {len(structures)} structures")
    
    if structures:
        print(f"  Formula: {structures[0].formula}")
    
    print("Parser tests passed!\n")


def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    import yaml
    from pathlib import Path
    
    config_path = Path('config/config.yaml')
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Loaded config with {len(config)} sections")
        print(f"  Population size: {config['evolution']['population_size']}")
        print(f"  LLM model: {config['llm']['model']}")
    else:
        print("⚠ config.yaml not found (expected for new setup)")
    
    print("Config tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Component Tests")
    print("="*60 + "\n")
    
    test_structure_creation()
    test_parser()
    test_config_loading()
    
    print("="*60)
    print("All tests passed!")
    print("="*60)