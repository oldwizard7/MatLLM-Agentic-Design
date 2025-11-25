"""Evolutionary algorithm for crystal structure search"""

from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import logging

from .structure import CrystalStructure

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """
    Evolutionary algorithm engine
    
    Implements:
    - Population initialization
    - Selection
    - Ranking
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evolution engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.population_size = config['evolution']['population_size']
        self.parent_size = config['evolution']['parent_size']
        
        # Selection strategy
        self.objective = config['objective']['type']
        self.target_ed = config['objective']['target_ed']
    
    def initialize_population(
        self,
        #需要data的地方
        data_path: str,
        filters: Optional[Dict] = None
    ) -> List[CrystalStructure]:
        """
        Initialize parent population from database
        
        Args:
            data_path: Path to structure database
            filters: Optional filters (e.g., element types, num_elements)
            
        Returns:
            Initial population of structures
        """
        logger.info(f"Initializing population from {data_path}")
        
        # Load structures from data path
        structures = self._load_structures(data_path, filters)
        
        # Sample initial population
        if len(structures) < self.population_size * self.parent_size:
            logger.warning(
                f"Not enough structures. Found {len(structures)}, "
                f"need {self.population_size * self.parent_size}"
            )
            return structures
        
        # Sort by stability (or other criteria)
        sorted_structures = sorted(
            structures,
            key=lambda s: s.decomposition_energy if s.decomposition_energy else float('inf')
        )
        
        # Select top structures
        selected = sorted_structures[:self.population_size * self.parent_size]
        
        logger.info(f"Selected {len(selected)} structures for initial population")
        
        return selected
    
    def _load_structures(
        self,
        data_path: str,
        filters: Optional[Dict]
    ) -> List[CrystalStructure]:
        """Load structures from database"""
        structures = []
        data_dir = Path(data_path)
        
        if not data_dir.exists():
            logger.error(f"Data path does not exist: {data_path}")
            return structures
        
        # Load POSCAR files
        for file_path in data_dir.glob("*.vasp"):
            try:
                with open(file_path, 'r') as f:
                    poscar_str = f.read()
                struct = CrystalStructure.from_poscar(poscar_str)
                
                # Apply filters
                if filters and not self._passes_filters(struct, filters):
                    continue
                
                structures.append(struct)
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # Load JSON files
        for file_path in data_dir.glob("*.json"):
            try:
                struct = CrystalStructure.load(str(file_path))
                
                if filters and not self._passes_filters(struct, filters):
                    continue
                
                structures.append(struct)
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(structures)} structures")
        return structures
    
    def _passes_filters(self, structure: CrystalStructure, filters: Dict) -> bool:
        """Check if structure passes filters"""
        # Number of elements
        if 'num_elements' in filters:
            min_elem, max_elem = filters['num_elements']
            num_unique = len(set(structure.species))
            if not (min_elem <= num_unique <= max_elem):
                return False
        
        # Exclude elements
        if 'exclude_elements' in filters:
            if any(elem in structure.species for elem in filters['exclude_elements']):
                return False
        
        # Include only certain elements
        if 'include_elements' in filters:
            if not all(elem in filters['include_elements'] for elem in structure.species):
                return False
        
        return True
    
    def select(
        self,
        parents: List[CrystalStructure],
        children: List[CrystalStructure],
        extra_pool: Optional[List[CrystalStructure]] = None
    ) -> List[CrystalStructure]:
        """
        Select next generation
        
        Args:
            parents: Current parent pool
            children: Newly generated children
            extra_pool: Optional extra structures
            
        Returns:
            Next generation parent pool
        """
        # Combine all candidates
        candidates = parents + children
        if extra_pool:
            candidates += extra_pool
        
        # Remove duplicates
        candidates = self._remove_duplicates(candidates)
        
        # Rank by objective
        ranked = self._rank_structures(candidates)
        
        # Select top K*P
        top_k = self.population_size * self.parent_size
        selected = ranked[:top_k]
        
        logger.info(
            f"Selected {len(selected)} structures from {len(candidates)} candidates"
        )
        
        return selected
    
    def _remove_duplicates(
        self,
        structures: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """Remove duplicate structures"""
        unique = []
        seen_formulas = set()
        
        for struct in structures:
            # Simple duplication check by formula
            # In production, use structure matcher
            if struct.formula not in seen_formulas:
                unique.append(struct)
                seen_formulas.add(struct.formula)
        
        return unique
    
    def _rank_structures(
        self,
        structures: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """
        Rank structures by objective
        
        Args:
            structures: List of structures to rank
            
        Returns:
            Sorted list (best first)
        """
        if self.objective == "stability":
            # Rank by decomposition energy (lower is better)
            return sorted(
                structures,
                key=lambda s: s.decomposition_energy if s.decomposition_energy is not None else float('inf')
            )
        
        elif self.objective == "bulk_modulus":
            # Rank by bulk modulus (higher is better)
            return sorted(
                structures,
                key=lambda s: s.properties.get('bulk_modulus', 0),
                reverse=True
            )
        
        elif self.objective == "multi_objective":
            # Weighted sum of objectives
            weights = self.config['objective']['weights']
            
            def multi_objective_score(s):
                # Normalize scores
                ed_score = s.decomposition_energy if s.decomposition_energy else float('inf')
                bm_score = s.properties.get('bulk_modulus', 0)
                
                # Combine (lower is better for ed, higher for bm)
                # Normalize bulk modulus to 0-1 range (assume max ~300 GPa)
                normalized_bm = min(bm_score / 300.0, 1.0)
                
                # Combined score (lower is better)
                score = weights['stability'] * ed_score - weights['property'] * normalized_bm
                return score
            
            return sorted(structures, key=multi_objective_score)
        
        else:
            logger.warning(f"Unknown objective: {self.objective}, using stability")
            return sorted(
                structures,
                key=lambda s: s.decomposition_energy if s.decomposition_energy else float('inf')
            )
    
    def calculate_diversity(self, structures: List[CrystalStructure]) -> float:
        """
        Calculate structural diversity metric
        
        Args:
            structures: List of structures
            
        Returns:
            Diversity score (0-1)
        """
        if len(structures) <= 1:
            return 0.0
        
        # Simple diversity: unique formulas / total structures
        unique_formulas = len(set(s.formula for s in structures))
        diversity = unique_formulas / len(structures)
        
        return diversity