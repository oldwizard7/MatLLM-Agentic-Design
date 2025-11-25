"""Structure Generator Agent - Generate structures from scratch"""

import logging
from typing import List, Dict
from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class StructureGeneratorAgent:
    """
    Generate new crystal structures from scratch
    
    Uses reference examples to show format, but generates entirely new structures.
    Does NOT modify existing structures (no crossover/mutation of parents).
    """
    
    def __init__(self, config: Dict):
        """
        Initialize structure generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm = LLMClient(config['llm'])
        
        # Reference examples (3-5 structures showing format)
        self.reference_examples = []
        
        # Success examples (previously generated good structures)
        self.success_examples = []
        self.max_success_examples = 10
    
    def set_reference_examples(self, examples: List[CrystalStructure]):
        """
        Set reference examples for format demonstration
        
        Args:
            examples: List of reference structures (3-5 examples)
        """
        self.reference_examples = examples
        logger.info(f"Set {len(examples)} reference examples for format")
    
    def add_success_example(self, structure: CrystalStructure):
        """
        Add a successful structure to example pool
        
        Args:
            structure: A successfully generated structure
        """
        self.success_examples.append(structure)
        
        # Keep pool size limited (only recent successes)
        if len(self.success_examples) > self.max_success_examples:
            self.success_examples.pop(0)
        
        logger.debug(f"Added success example: {structure.formula}")
    
    def generate(
        self,
        task_description: str,
        constraints: Dict,
        n_structures: int = 5,
        strategy: str = None
    ) -> List[str]:
        """
        Generate new structures from scratch

        Args:
            task_description: Description of what to generate
                             e.g., "stable wide-bandgap semiconductors"
            constraints: Property constraints
                        e.g., {'band_gap': '>2.5 eV', 'formation_energy': '<-1.0 eV/atom'}
            n_structures: Number of structures to generate
            strategy: Search strategy from curate phase (optional)
                     e.g., "Focus on exploring lower energy configurations"

        Returns:
            List of generated structures as POSCAR strings
        """
        logger.info(f"Generating {n_structures} new structures for: {task_description}")
        if strategy:
            logger.info(f"Using strategy: {strategy[:100]}...")

        # Build prompt
        prompt = self._build_generation_prompt(
            task_description,
            constraints,
            n_structures,
            strategy
        )
        
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Generate using LLM
        try:
            responses = self.llm.generate(prompt, n=1)
            logger.info(f"LLM generated {len(responses)} response(s)")
            return responses
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return []
    
    def _build_generation_prompt(
        self,
        task_desc: str,
        constraints: Dict,
        n: int,
        strategy: str = None
    ) -> str:
        """
        Build prompt for structure generation

        Args:
            task_desc: Task description
            constraints: Property constraints
            n: Number of structures to generate
            strategy: Search strategy from curate phase (optional)

        Returns:
            Complete prompt string
        """
        prompt_parts = [
            "You are an expert materials scientist with deep knowledge of crystal structures.",
            "",
            f"TASK: Generate {n} ENTIRELY NEW crystal structures for: {task_desc}",
            "",
            "IMPORTANT:",
            "- Generate COMPLETELY NEW structures (do NOT copy the examples)",
            "- Use CREATIVE element combinations",
            "- Ensure structures are CHEMICALLY PLAUSIBLE",
            "- Follow the POSCAR format exactly as shown",
            "",
            "CONSTRAINTS:"
        ]
        
        # Add constraints
        for key, value in constraints.items():
            if value is not None:
                prompt_parts.append(f"  - {key}: {value}")

        # Add strategy guidance from curate phase (KEY ADDITION!)
        if strategy:
            prompt_parts.extend([
                "",
                "="*60, #分隔线而已
                "SEARCH STRATEGY (Based on previous iteration feedback):",
                "="*60,
                "",
                strategy,
                "",
                "APPLY THE ABOVE STRATEGY when generating new structures.",
                ""
            ])

        prompt_parts.extend([
            "",
            "="*60,
            "REFERENCE EXAMPLES (Format demonstration - DO NOT COPY):",
            "="*60,
            ""
        ])
        
        # Add reference examples (format demonstration)
        for i, ref in enumerate(self.reference_examples[:3], 1):
            prompt_parts.extend([
                f"Example {i}: {ref.formula}",
                ref.to_poscar(),
                ""
            ])
        
        # Add success examples if available 之前的成功的example对于下一次循环的作用
        if self.success_examples:
            prompt_parts.extend([
                "="*60,
                "PREVIOUSLY SUCCESSFUL STRUCTURES (for inspiration only):",
                "="*60,
                ""
            ])
            
            for i, succ in enumerate(self.success_examples[-3:], 1):
                ed = succ.decomposition_energy if succ.decomposition_energy else "N/A"
                prompt_parts.extend([
                    f"Success {i}: {succ.formula} (Ed = {ed})",
                    succ.to_poscar(),
                    ""
                ])
        
        # Add generation instructions
        prompt_parts.extend([
            "="*60,
            "YOUR TASK:",
            "="*60,
            "",
            f"Generate {n} ENTIRELY NEW crystal structures that:",
            "1. Satisfy all the constraints above",
            "2. Use DIFFERENT element combinations",
            "3. Are chemically realistic and synthesizable",
            "4. Follow the exact POSCAR format shown",
            "",
            "For each structure, output in this format:",
            "",
            "Structure 1:",
            "<formula>",
            "1.0",
            "<lattice parameters: 3 lines of 3 numbers each>",
            "<element symbols>",
            "<number of each element>",
            "direct",
            "<fractional coordinates with element labels>",
            "",
            "Structure 2:",
            "...",
            "",
            f"Generate exactly {n} different structures now:",
            ""
        ])
        
        return "\n".join(prompt_parts)


# Convenience function for testing
def test_generator():
    """Test the structure generator"""
    config = {
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4',
            'temperature': 0.9,
            'max_tokens': 4096
        }
    }
    
    generator = StructureGeneratorAgent(config)
    
    # Create dummy reference
    from src.core.structure import CrystalStructure
    import numpy as np
    
    ref = CrystalStructure(
        formula="NaCl",
        lattice=np.eye(3) * 5.64,
        positions=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
        species=['Na', 'Cl']
    )
    
    generator.set_reference_examples([ref])
    
    # Generate
    results = generator.generate(
        task_description="stable wide-bandgap semiconductors",
        constraints={'band_gap': '>2.5 eV', 'formation_energy': '<-1.0 eV/atom'},
        n_structures=2
    )
    
    print(f"Generated {len(results)} structures")
    for i, result in enumerate(results):
        print(f"\nStructure {i+1}:")
        print(result[:200] + "...")


if __name__ == "__main__":
    test_generator()