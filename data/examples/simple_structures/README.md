# Example Structures

This directory contains simple example structures for testing MatLLMSearch.

## Structures

1. **NaCl.json** - Sodium Chloride (Rock Salt)
   - Space group: Fm-3m (225)
   - Lattice: FCC
   - Common ionic compound

2. **CsCl.json** - Cesium Chloride
   - Space group: Pm-3m (221)
   - Lattice: Simple cubic
   - Alternative ionic structure

3. **CaTiO3.json** - Calcium Titanate (Perovskite)
   - Space group: Pm-3m (221)
   - Lattice: Cubic perovskite
   - Oxide structure

## Usage
```python
from src.utils.data_manager import DataManager

dm = DataManager()
structures = dm.load_structures('data/examples/simple_structures/')
```
