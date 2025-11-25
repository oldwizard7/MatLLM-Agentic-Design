"""Orchestrator Agent - ACE loop controller"""

from typing import List, Dict, Optional
import logging
import numpy as np
import yaml

from src.core.structure import CrystalStructure
from src.core.evolution import EvolutionEngine
from src.agents.hypothesis_gen import HypothesisGenerationAgent
from src.agents.workers import WorkerAgents

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Orchestrator Agent - Strategic leader of the search system
    
    Implements ACE (Agentic Context Engineering):
    - Generate: Execute current search phase
    - Reflect: Analyze results
    - Curate: Evolve strategy
    """
    
    def __init__(self, config: Dict, prompts: Dict):
        """
        Initialize orchestrator
        
        Args:
            config: Configuration dictionary
            prompts: Prompt templates
        """
        self.config = config
        self.prompts = prompts
        
        # Sub-agents
        # 可实现ReAct
        self.hypothesis_agent = HypothesisGenerationAgent(config, prompts)
        # Worker agents
        self.worker_agents = WorkerAgents(config)
        
        # Structure generator (generate from scratch, not modify)
        from src.agents.workers.structure_generator import StructureGeneratorAgent
        self.structure_generator = StructureGeneratorAgent(config)

        # Tool-based generation system
        from src.core.operators import StructureOperators
        from src.agents.tool_selector import ToolSelectionAgent
        self.operators = StructureOperators()
        self.tool_selector = ToolSelectionAgent(config)

        self.evolution = EvolutionEngine(config)


        # Search state
        self.current_plan = None
        self.search_history = []
        self.iteration = 0
        
        # Budget tracking
        self.budget_remaining = config.get('budget', float('inf'))
    
    def initialize(self, reference_path: str, filters: Optional[Dict] = None) -> List[CrystalStructure]:
        """
        Initialize search with reference examples

        Args:
            reference_path: Path to reference examples (3-5 structures for format)
            filters: Not used (kept for compatibility)

        Returns:
            Empty example pool (will be filled during search)
        """
        logger.info("Initializing search...")
        logger.info(f"Loading reference examples from {reference_path}")

        # Load reference examples
        from src.utils.data_manager import DataManager
        dm = DataManager()
        reference_examples = dm.load_structures(reference_path)

        if not reference_examples:
            logger.error(f"No reference examples found in {reference_path}")
            raise ValueError("Need at least 1 reference example")

        logger.info(f"Loaded {len(reference_examples)} reference examples:")
        for ref in reference_examples:
            logger.info(f"  - {ref.formula}")

        # Set reference examples for generator
        self.structure_generator.set_reference_examples(reference_examples)

        # Start with empty example pool (will be populated with successful generations)
        example_pool: List[CrystalStructure] = []

        logger.info("Initialization complete. Ready to generate structures.")

        return example_pool
    
    def generate(self, example_pool: List[CrystalStructure]) -> List[CrystalStructure]:
        """
        Generate: Create new structures using tool-based approach

        Args:
            example_pool: Previously successful structures

        Returns:
            Newly generated structures
        """
        logger.info(f"=== Generate Phase (Iteration {self.iteration}) ===")

        # Get task description and constraints
        task_desc = self.config.get('objective', {}).get('description', 'stable materials')

        # Build constraints dict
        constraints = {}
        obj_config = self.config.get('objective', {})

        if 'band_gap' in obj_config:
            constraints['band_gap'] = f">{obj_config['band_gap']} eV"

        if 'formation_energy' in obj_config:
            constraints['formation_energy'] = f"<{obj_config['formation_energy']} eV/atom"

        if 'target_ed' in obj_config:
            constraints['decomposition_energy'] = f"<{obj_config['target_ed']} eV/atom"

        logger.info(f"Task: {task_desc}")
        logger.info(f"Constraints: {constraints}")

        # Get number of structures to generate
        n_structures = self.config.get('evolution', {}).get('children_size', 5)
        logger.info(f"Generating {n_structures} new structures...")

        # Get previous reflection for tool selection
        prev_reflection = self.search_history[-1] if self.search_history else {}

        # ===== TOOL-BASED GENERATION =====
        # Step 1: Agent decides which tools to use
        logger.info("Tool selection agent analyzing state...")
        tool_actions = self.tool_selector.select_tools(
            example_pool=example_pool,
            strategy=self.current_plan,
            reflection=prev_reflection,
            n_structures=n_structures
        )

        # Step 2: Execute tool actions
        logger.info(f"Executing {len(tool_actions)} tool actions...")
        children = []

        for i, action in enumerate(tool_actions):
            tool_name = action['tool']
            logger.info(f"  Action {i+1}/{len(tool_actions)}: {tool_name}")

            try:
                if tool_name == 'substitute':
                    # Element substitution
                    parent = example_pool[action['parent']]
                    child = self.operators.substitute_element(
                        parent=parent,
                        old_element=action['old_element'],
                        new_element=action['new_element']
                    )
                    children.append(child)

                elif tool_name == 'mutate':
                    # Structure mutation
                    parent = example_pool[action['parent']]
                    child = self.operators.mutate_structure(
                        parent=parent,
                        strength=action['strength']
                    )
                    children.append(child)

                elif tool_name == 'mix':
                    # Mix two parents
                    parent1 = example_pool[action['parent1']]
                    parent2 = example_pool[action['parent2']]
                    child = self.operators.mix_structures(
                        parent1=parent1,
                        parent2=parent2,
                        ratio=action['ratio']
                    )
                    children.append(child)

                elif tool_name == 'new':
                    # Generate from scratch using LLM
                    logger.info("    Calling LLM for fresh generation...")

                    # Update success examples
                    for struct in example_pool[-5:]:
                        self.structure_generator.add_success_example(struct)

                    # Generate using LLM
                    generated_texts = self.structure_generator.generate(
                        task_description=task_desc,
                        constraints=constraints,
                        n_structures=1,  # One at a time
                        strategy=self.current_plan
                    )

                    if generated_texts:
                        # Parse generated structure
                        from src.utils.parser import StructureParser
                        parsed = StructureParser.parse_structures(generated_texts[0])
                        children.extend(parsed)
                    else:
                        logger.warning("    LLM generation failed")

            except Exception as e:
                logger.error(f"  Tool execution failed: {e}")
                continue

        logger.info(f"Generated {len(children)} structures from tools")

        # Step 3: Evaluate structures
        if children:
            logger.info("Evaluating structures...")
            from src.core.evaluator import StructureEvaluator
            evaluator = StructureEvaluator(self.config)

            evaluated = []
            for i, child in enumerate(children):
                logger.info(f"  Evaluating {i+1}/{len(children)}: {child.formula}")
                try:
                    evaluated_child = evaluator.evaluate(child)
                    evaluated.append(evaluated_child)
                except Exception as e:
                    logger.warning(f"  Evaluation failed for {child.formula}: {e}")
                    child.is_valid = False
                    evaluated.append(child)

            children = evaluated

        logger.info(f"Generated {len(children)} structures in this iteration")

        return children
    
    def reflect(self, children: List[CrystalStructure]) -> Dict:
        """
        Reflect: Analyze results
        
        Args:
            children: Generated child structures
            
        Returns:
            Reflection dictionary with statistics
        """
        logger.info("=== Reflect Phase ===")
        
        # Calculate statistics
        valid_children = [c for c in children if c.is_valid]
        metastable_children = [c for c in valid_children 
                               if c.decomposition_energy is not None 
                               and c.decomposition_energy < 0.1]
        stable_children = [c for c in metastable_children 
                          if c.decomposition_energy < 0]
        
        reflection = {
            'iteration': self.iteration,
            'total_generated': len(children),
            'valid_count': len(valid_children),
            'metastable_count': len(metastable_children),
            'stable_count': len(stable_children),
            'valid_rate': 100 * len(valid_children) / len(children) if children else 0,
            'metastable_rate': 100 * len(metastable_children) / len(children) if children else 0,
            'stable_rate': 100 * len(stable_children) / len(children) if children else 0
        }
        
        # Energy statistics
        if metastable_children:
            energies = [c.decomposition_energy for c in metastable_children]
            reflection['best_ed'] = min(energies)
            reflection['avg_ed'] = np.mean(energies)
            reflection['worst_ed'] = max(energies)
        else:
            reflection['best_ed'] = float('inf')
            reflection['avg_ed'] = float('inf')
            reflection['worst_ed'] = float('inf')
        
        # Diversity
        reflection['diversity'] = self.evolution.calculate_diversity(valid_children)
        
        # Log summary
        logger.info(f"Valid: {reflection['valid_rate']:.1f}%")
        logger.info(f"Metastable: {reflection['metastable_rate']:.1f}%")
        logger.info(f"Best Ed: {reflection['best_ed']:.3f} eV/atom")
        logger.info(f"Diversity: {reflection['diversity']:.2f}")
        
        # Store in history
        self.search_history.append(reflection)
        
        return reflection
    
    def curate(self, reflection: Dict, example_pool: List[CrystalStructure]) -> str:
        """
        Curate: Evolve strategy using hypothesis agent

        Args:
            reflection: Performance reflection
            example_pool: Current example pool (parent structures)

        Returns:
            Updated search plan as a formatted string
        """
        logger.info("=== Curate Phase ===")

        # Use hypothesis agent to generate strategy (ReAct approach)
        strategy_dict = self.hypothesis_agent.propose_strategy(
            parent_pool=example_pool,
            current_plan=self.current_plan,
            reflection=reflection
        )

        # Format strategy dictionary into a readable string for LLM prompt
        strategy_text = self._format_strategy(strategy_dict)

        logger.info(f"New strategy generated:")
        logger.info(f"  Thought: {strategy_dict['thought'][:80]}...")
        logger.info(f"  Action: {strategy_dict['action']}")

        self.current_plan = strategy_text

        return strategy_text

    def _format_strategy(self, strategy_dict: Dict) -> str:
        """
        Format strategy dictionary into readable text for LLM prompt

        Args:
            strategy_dict: Strategy from hypothesis agent

        Returns:
            Formatted strategy text
        """
        parts = [
            "REASONING:",
            f"{strategy_dict['thought']}",
            "",
            "RECOMMENDED ACTION:",
            f"{strategy_dict['action']}",
            "",
            "EXPECTED OUTCOME:",
            f"{strategy_dict['expected_outcome']}"
        ]

        return "\n".join(parts)
    
    def should_terminate(self, reflection: Dict) -> bool:
        """
        Check termination conditions
        
        Args:
            reflection: Current iteration reflection
            
        Returns:
            True if search should terminate
        """
        # Budget exhausted
        if self.budget_remaining <= 0:
            logger.info("Budget exhausted")
            return True
        
        # Target achieved
        if reflection['best_ed'] < self.config['objective']['target_ed']:
            logger.info(f"Target achieved: {reflection['best_ed']:.3f} < {self.config['objective']['target_ed']}")
            return True
        
        # Max iterations reached
        max_iter = self.config['evolution']['max_iterations']
        if self.iteration >= max_iter:
            logger.info(f"Max iterations reached: {self.iteration} >= {max_iter}")
            return True
        
        return False
    
    def save_checkpoint(self, parent_pool: List[CrystalStructure], output_dir: str):
        """Save search checkpoint"""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save structures
        for i, struct in enumerate(parent_pool):
            struct.save(str(output_path / f"structure_{self.iteration}_{i}.json"))
        
        # Save history
        # ---- Fix numpy.float32 serialization ----
        def _safe(x):
            import numpy as np
            if isinstance(x, (np.floating, np.float32, np.float64)):
                return float(x)
            if hasattr(x, "item"):  # numpy scalar or torch tensor
                return float(x.item())
            return x

        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return _safe(obj)

        clean_history = _sanitize(self.search_history)

        with open(output_path / f"history_{self.iteration}.json", 'w') as f:
            json.dump(clean_history, f, indent=2)
        # ---- End fix ----

        
        logger.info(f"Checkpoint saved to {output_dir}")