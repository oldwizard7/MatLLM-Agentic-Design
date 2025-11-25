"""Hypothesis Generation Agent - Strategy optimization using GEPA"""
import numpy as np
from typing import List, Dict, Optional
import logging

from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class HypothesisGenerationAgent:
    """
    Hypothesis Generation Agent
    
    Implements GEPA (Genetic Evolution of Prompt Alternatives)
    to optimize search strategies
    """
    
    def __init__(self, config: Dict, prompts: Dict):
        """
        Initialize hypothesis generation agent
        
        Args:
            config: Configuration dictionary
            prompts: Prompt templates dictionary
        """
        self.config = config
        self.prompts = prompts
        self.llm = LLMClient(config['llm'])
        
        # Strategy history for evolution
        self.strategy_history = []
        self.performance_history = []
    
    def propose_strategy(
        self,
        parent_pool: List[CrystalStructure],
        #存在这么一个current_plan参数
        current_plan: Optional[str] = None,
        reflection: Optional[Dict] = None
    ) -> Dict:
        """
        Propose search strategy for current iteration
        
        Uses ReAct (Reasoning-Action-Observation) pattern
        
        Args:
            parent_pool: Current parent structures
            current_plan: Current search plan
            reflection: Performance reflection from previous iteration
            
        Returns:
            Strategy dictionary with prompt template and parameters
        """
        logger.info("Generating hypothesis for next iteration")
        
        # Analyze current state
        analysis = self._analyze_state(parent_pool, reflection)
        
        # Generate strategy using ReAct
        strategy = self._react_loop(analysis, current_plan)
        
        # Store for evolution
        self.strategy_history.append(strategy)
        
        return strategy
    
    def _analyze_state(
        self,
        parents: List[CrystalStructure],
        reflection: Optional[Dict]
    ) -> Dict:
        """
        Analyze current search state
        
        Returns:
            Analysis dictionary with key observations
        """
        num_parents = len(parents)

        if num_parents == 0:
            analysis = {
                'num_parents': 0,
                'best_ed': None,
                'avg_ed': None,
                'composition_diversity': 0.0,
            }
        else:
            valid_eds = [
                p.decomposition_energy for p in parents
                if getattr(p, 'decomposition_energy', None) is not None
            ]

            best_ed = min(valid_eds) if valid_eds else None
            avg_ed = float(np.mean(valid_eds)) if valid_eds else None

            composition_diversity = (
                len(set(getattr(p, 'formula', None) for p in parents)) / num_parents
                if num_parents else 0.0
            )

            analysis = {
                'num_parents': num_parents,
                'best_ed': best_ed,
                'avg_ed': avg_ed,
                'composition_diversity': composition_diversity,
            }
        
        if reflection:
            analysis['previous_valid_rate'] = reflection.get('valid_rate', 0)
            analysis['previous_metastable_rate'] = reflection.get('metastable_rate', 0)
        
        return analysis
    
    def _react_loop(self, analysis: Dict, current_plan: Optional[str]) -> Dict:
        """
        ReAct reasoning loop
        
        Thought → Action → Observation
        
        Args:
            analysis: Current state analysis
            current_plan: Current strategy
            
        Returns:
            New strategy
        """
        # Thought: Analyze what needs improvement
        thought = self._generate_thought(analysis, current_plan)
        logger.info(f"Thought: {thought}")
        
        # Action: Propose modification strategy
        action = self._generate_action(thought, analysis)
        logger.info(f"Action: {action}")
        
        # Observation: Expected outcome (simulated)
        observation = self._generate_observation(action, analysis)
        logger.info(f"Expected outcome: {observation}")
        
        # Compile into strategy
        strategy = {
            'thought': thought,
            'action': action,
            'expected_outcome': observation,
            'prompt_template': self._select_prompt_template(action),
            'template_vars': self._generate_template_vars(action, analysis)
        }
        
        return strategy
    
    def _generate_thought(self, analysis: Dict, current_plan: Optional[str]) -> str:
        """Generate reasoning about current state"""
        best_ed = analysis.get('best_ed')
        avg_ed = analysis.get('avg_ed')

        if best_ed is None:
            return ("Insufficient energy data from parents. "
                   "Focus on generating diverse candidates and obtaining valid energies.")
        if best_ed > 0.1:
            return (f"Current structures are far from stability (best Ed = {best_ed:.3f}). "
                   f"Need to explore lower energy configurations.")
        elif best_ed > 0:
            return (f"Close to stability (best Ed = {best_ed:.3f}). "
                   f"Need fine-tuning to reach convex hull.")
        else:
            return (f"Achieved stable structures (best Ed = {best_ed:.3f}). "
                   f"Can optimize for additional properties.")
    
    def _generate_action(self, thought: str, analysis: Dict) -> str:
        """Generate action based on thought"""
        best_ed = analysis.get('best_ed')

        if best_ed is None:
            return "Generate diverse initial structures to gather energy estimates"
        if best_ed > 0.1:
            return "Apply isotropic compression to increase cohesive energy"
        elif best_ed > 0:
            return "Apply local perturbations to optimize atomic positions"
        else:
            return "Explore compositional substitutions for property enhancement"
    
    def _generate_observation(self, action: str, analysis: Dict) -> str:
        """Generate expected observation"""
        return f"Expected to generate structures with improved stability based on {action}"
    
    def _select_prompt_template(self, action: str) -> str:
        """Select appropriate prompt template based on action"""
        objective = self.config['objective']['type']
        
        if 'compression' in action.lower():
            return self.prompts.get('csg_gepa_optimized', self.prompts['csg_basic'])
        elif 'multi' in objective or 'property' in action.lower():
            return self.prompts.get('multi_objective', self.prompts['csg_basic'])
        else:
            return self.prompts['csg_basic']
    
    def _generate_template_vars(self, action: str, analysis: Dict) -> Dict:
        """Generate template variables for prompt"""
        return {
            'target_property': self.config['objective']['type'],
            'target_value': self.config['objective']['target_ed'],
            'current_best': analysis['best_ed'],
            'observation': action
        }
    
    def evolve_prompts(self, performance_data: Dict):
        """
        Evolve prompts based on performance (GEPA)
        
        This is a placeholder for genetic algorithm on prompts
        
        Args:
            performance_data: Performance metrics
        """
        self.performance_history.append(performance_data)
        
        # TODO: Implement genetic algorithm for prompt evolution
        # - Selection: Choose best performing prompts
        # - Crossover: Combine elements from successful prompts
        # - Mutation: Modify prompt components
        
        logger.info("Prompt evolution not yet implemented")
