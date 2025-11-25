"""Structure operators for material generation

Provides atomic operations for structure manipulation:
- Substitution: Replace elements
- Mutation: Perturb structure
- Mix: Combine parent structures
"""

import numpy as np
import logging
from typing import Optional
from src.core.structure import CrystalStructure

logger = logging.getLogger(__name__)


class StructureOperators:
    """
    Atomic operations for structure generation

    These are the "tools" that agents can call to generate new structures
    """

    @staticmethod
    def substitute_element(
        parent: CrystalStructure,
        old_element: str,
        new_element: str
    ) -> CrystalStructure:
        """
        Substitute an element in a structure

        Args:
            parent: Parent structure
            old_element: Element to replace
            new_element: Replacement element

        Returns:
            New structure with substituted element

        Example:
            ZnO + substitute(O→S) → ZnS
        """
        logger.info(f"Substituting {old_element} → {new_element} in {parent.formula}")

        # Replace element in species list
        new_species = [new_element if s == old_element else s for s in parent.species]

        # Update formula
        new_formula = parent.formula.replace(old_element, new_element)

        # Create new structure
        child = CrystalStructure(
            formula=new_formula,
            lattice=parent.lattice.copy(),
            positions=parent.positions.copy(),
            species=new_species
        )

        # Track provenance
        child.metadata['source'] = 'substitute'
        child.metadata['parent'] = parent.formula
        child.metadata['substitution'] = f"{old_element}→{new_element}"

        return child

    @staticmethod
    def mutate_structure(
        parent: CrystalStructure,
        strength: float = 0.1
    ) -> CrystalStructure:
        """
        Mutate a structure with random perturbations

        Args:
            parent: Parent structure
            strength: Mutation strength (0.05-0.2 recommended)
                     Controls magnitude of random changes

        Returns:
            Mutated structure

        Example:
            NaCl + mutate(0.1) → NaCl' (slightly perturbed)
        """
        logger.info(f"Mutating {parent.formula} with strength={strength}")

        # Perturb atomic positions
        new_positions = parent.positions.copy()
        perturbation = np.random.uniform(-strength, strength, new_positions.shape)
        new_positions += perturbation

        # Wrap to unit cell [0, 1)
        new_positions = np.mod(new_positions, 1.0)

        # Optionally perturb lattice (smaller magnitude)
        new_lattice = parent.lattice.copy()
        if np.random.random() < 0.5:  # 50% chance to also perturb lattice
            lattice_scale = 1.0 + np.random.uniform(-strength/2, strength/2)
            new_lattice *= lattice_scale

        # Create new structure
        child = CrystalStructure(
            formula=parent.formula,
            lattice=new_lattice,
            positions=new_positions,
            species=parent.species.copy()
        )

        # Track provenance
        child.metadata['source'] = 'mutate'
        child.metadata['parent'] = parent.formula
        child.metadata['mutation_strength'] = strength

        return child

    @staticmethod
    def mix_structures(
        parent1: CrystalStructure,
        parent2: CrystalStructure,
        ratio: float = 0.5
    ) -> CrystalStructure:
        """
        Mix two parent structures

        Args:
            parent1: First parent
            parent2: Second parent
            ratio: Mixing ratio (0-1)
                  0 = 100% parent2, 1 = 100% parent1

        Returns:
            Mixed structure

        Example:
            LiCoO2 + LiNiO2 + mix(0.6) → Li(Co0.6Ni0.4)O2
        """
        logger.info(f"Mixing {parent1.formula} × {parent2.formula} (ratio={ratio:.2f})")

        # Interpolate lattice parameters
        new_lattice = ratio * parent1.lattice + (1 - ratio) * parent2.lattice

        # Use parent1's structure as template
        # (More sophisticated: could mix compositions)
        new_positions = parent1.positions.copy()
        new_species = parent1.species.copy()

        # Generate mixed formula
        new_formula = f"{parent1.formula}_{ratio:.1f}_{parent2.formula}_{1-ratio:.1f}"

        # Create new structure
        child = CrystalStructure(
            formula=new_formula,
            lattice=new_lattice,
            positions=new_positions,
            species=new_species
        )

        # Track provenance
        child.metadata['source'] = 'mix'
        child.metadata['parent1'] = parent1.formula
        child.metadata['parent2'] = parent2.formula
        child.metadata['mix_ratio'] = ratio

        return child
