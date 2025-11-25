"""Simple demonstration of MatLLMSearch workflow"""

import sys
sys.path.insert(0, '.')

import logging
from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient
from src.core.evaluator import StructureEvaluator
from src.utils.parser import StructureParser as ParserUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_demo():
    """
    Simplified demo: One iteration of structure generation
    """
    logger.info("="*60)
    logger.info("MatLLMSearch Simple Demo")
    logger.info("="*60)
    
    # 1. Create mock parent structures (NaCl as example)
    logger.info("\n1. Creating parent structures...")
    
    parent_poscar = """NaCl
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
    
    parent = CrystalStructure.from_poscar(parent_poscar)
    logger.info(f"Parent structure: {parent.formula}")
    
    # 2. Generate new structures with LLM (mock for demo)
    logger.info("\n2. Generating new structures...")
    logger.info("(In real use, this calls OpenAI/Anthropic API)")
    
    # Mock LLM response
    mock_response = """Here are 2 new structures:

Structure 1:
NaCl
1.0
5.50 0.0 0.0
0.0 5.50 0.0
0.0 0.0 5.50
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

Structure 2:
KCl
1.0
6.28 0.0 0.0
0.0 6.28 0.0
0.0 0.0 6.28
K Cl
4 4
direct
0.0 0.0 0.0 K
0.5 0.5 0.0 K
0.5 0.0 0.5 K
0.0 0.5 0.5 K
0.5 0.5 0.5 Cl
0.0 0.0 0.5 Cl
0.0 0.5 0.0 Cl
0.5 0.0 0.0 Cl"""
    
    # 3. Parse structures
    logger.info("\n3. Parsing structures...")
    children = ParserUtils.parse_structures(mock_response)
    logger.info(f"Parsed {len(children)} structures")
    
    for i, child in enumerate(children):
        logger.info(f"  Child {i+1}: {child.formula}")
    
    # 4. Evaluate (requires CHGNet - skip if not installed)
    logger.info("\n4. Evaluation (skipped in demo)")
    logger.info("(In real use, this runs CHGNet relaxation and energy calculation)")
    
    logger.info("\n" + "="*60)
    logger.info("Demo complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Set up API keys (OPENAI_API_KEY)")
    logger.info("2. Install CHGNet: pip install chgnet")
    logger.info("3. Run full search: python src/main.py")


if __name__ == "__main__":
    simple_demo()