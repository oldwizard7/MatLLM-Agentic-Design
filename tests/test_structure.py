"""Unit tests for CrystalStructure"""

import pytest
import numpy as np
from src.core.structure import CrystalStructure


class TestCrystalStructure:
    """Test CrystalStructure class"""
    
    def test_creation(self):
        """Test basic structure creation"""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ['Na', 'Cl']
        
        struct = CrystalStructure(
            formula="NaCl",
            lattice=lattice,
            positions=positions,
            species=species
        )
        
        assert struct.formula == "NaCl"
        assert struct.num_atoms == 2
        assert np.isclose(struct.volume, 125.0)
    
    def test_poscar_conversion(self):
        """Test POSCAR to/from conversion"""
        poscar_str = """NaCl
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
0.5 0.0 0.0 Cl"""
        
        struct = CrystalStructure.from_poscar(poscar_str)
        assert struct.formula == "NaCl"
        assert struct.num_atoms == 8
        
        # Convert back
        new_poscar = struct.to_poscar()
        struct2 = CrystalStructure.from_poscar(new_poscar)
        
        assert struct2.formula == struct.formula
        assert struct2.num_atoms == struct.num_atoms
        assert np.allclose(struct2.lattice, struct.lattice)
    
    def test_serialization(self):
        """Test JSON serialization"""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0]])
        species = ['Na']
        
        struct = CrystalStructure(
            formula="Na",
            lattice=lattice,
            positions=positions,
            species=species
        )
        
        # Convert to dict
        data = struct.to_dict()
        assert data['formula'] == "Na"
        
        # Convert back
        struct2 = CrystalStructure.from_dict(data)
        assert struct2.formula == struct.formula
        assert np.allclose(struct2.lattice, struct.lattice)
    
    def test_invalid_creation(self):
        """Test that invalid inputs raise errors"""
        with pytest.raises(ValueError):
            # Wrong lattice shape
            CrystalStructure(
                formula="Na",
                lattice=np.eye(2),  # Should be 3x3
                positions=np.array([[0.0, 0.0, 0.0]]),
                species=['Na']
            )
        
        with pytest.raises(ValueError):
            # Mismatched positions and species
            CrystalStructure(
                formula="Na",
                lattice=np.eye(3),
                positions=np.array([[0.0, 0.0, 0.0]]),
                species=['Na', 'Cl']  # Two species but one position
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])