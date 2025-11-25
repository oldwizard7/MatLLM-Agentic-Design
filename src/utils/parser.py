"""Utilities for parsing LLM outputs"""

import re
import logging
from typing import List, Optional

from src.core.structure import CrystalStructure

logger = logging.getLogger(__name__)


class StructureParser:
    """Parse LLM-generated structure strings"""
    
    @staticmethod
    def extract_poscar_blocks(text: str) -> List[str]:
        """
        Extract POSCAR blocks from LLM response
        
        Args:
            text: LLM response text
            
        Returns:
            List of POSCAR strings
        """
        # Pattern 1: Look for typical POSCAR structure
        # Comment line + scaling + 3 lattice vectors + species + counts + direct + positions
        
        poscar_blocks = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            # Check if this looks like start of POSCAR
            if i + 8 < len(lines):
                # Try to parse as POSCAR
                block_lines = []
                
                # Comment line
                block_lines.append(lines[i])
                i += 1
                
                # Check for scaling factor (should be a number)
                try:
                    float(lines[i].strip())
                    block_lines.append(lines[i])
                    i += 1
                except:
                    continue
                
                # Try to get 3 lattice vectors
                lattice_ok = True
                for _ in range(3):
                    parts = lines[i].split()
                    if len(parts) == 3:
                        try:
                            [float(x) for x in parts]
                            block_lines.append(lines[i])
                            i += 1
                        except:
                            lattice_ok = False
                            break
                    else:
                        lattice_ok = False
                        break
                
                if not lattice_ok:
                    continue
                
                # Species line
                block_lines.append(lines[i])
                i += 1
                
                # Counts line
                block_lines.append(lines[i])
                i += 1
                
                # Coordinate type
                block_lines.append(lines[i])
                i += 1
                
                # Positions (until we hit a blank line or non-numeric)
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        break
                    
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            [float(x) for x in parts[:3]]
                            block_lines.append(lines[i])
                            i += 1
                        except:
                            break
                    else:
                        break
                
                # Validate we got a complete POSCAR
                if len(block_lines) >= 10:
                    poscar_blocks.append('\n'.join(block_lines))
            else:
                i += 1
        
        return poscar_blocks
    
    @staticmethod
    def parse_structures(text: str) -> List[CrystalStructure]:
        """
        Parse crystal structures from LLM response
        
        Args:
            text: LLM response text
            
        Returns:
            List of parsed CrystalStructure objects
        """
        structures = []
        
        # Extract POSCAR blocks
        poscar_blocks = StructureParser.extract_poscar_blocks(text)
        
        logger.info(f"Found {len(poscar_blocks)} POSCAR blocks in response")
        
        # Parse each block
        for i, poscar_str in enumerate(poscar_blocks):
            try:
                struct = CrystalStructure.from_poscar(poscar_str)
                structures.append(struct)
                logger.debug(f"Successfully parsed structure {i+1}: {struct.formula}")
            except Exception as e:
                logger.warning(f"Failed to parse POSCAR block {i+1}: {e}")
        
        return structures
    
    @staticmethod
    def validate_structure_string(poscar_str: str) -> bool:
        """
        Quick validation of POSCAR string format
        
        Args:
            poscar_str: POSCAR string
            
        Returns:
            True if valid format
        """
        try:
            lines = [l.strip() for l in poscar_str.split('\n') if l.strip()]
            
            # Minimum lines needed
            if len(lines) < 9:
                return False
            
            # Line 2 should be scaling factor
            float(lines[1])
            
            # Lines 3-5 should be lattice vectors
            for i in range(2, 5):
                parts = lines[i].split()
                if len(parts) != 3:
                    return False
                [float(x) for x in parts]
            
            # Line 7 should be counts (integers)
            counts = [int(x) for x in lines[6].split()]
            
            # Line 8 should be coordinate type
            coord_type = lines[7].lower()
            if not (coord_type.startswith('d') or coord_type.startswith('c')):
                return False
            
            # Should have positions
            total_atoms = sum(counts)
            if len(lines) < 8 + total_atoms:
                return False
            
            return True
            
        except Exception:
            return False