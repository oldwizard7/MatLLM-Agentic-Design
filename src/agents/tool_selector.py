"""Tool Selection Agent - Decides which tools to use for structure generation

Uses LLM to intelligently select and parameterize tools based on:
- Current search strategy
- Parent structures
- Historical performance
"""

import logging
import json
import re
from typing import List, Dict, Optional
from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ToolSelectionAgent:
    """
    Agent that decides which tools to use for generation

    Implements intelligent tool selection using LLM reasoning
    """

    def __init__(self, config: Dict):
        """
        Initialize tool selection agent

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm = LLMClient(config['llm'])

    def select_tools(
        self,
        example_pool: List[CrystalStructure],
        strategy: Optional[str],
        reflection: Dict,
        n_structures: int
    ) -> List[Dict]:
        """
        Select tools and parameters for structure generation

        Args:
            example_pool: Available parent structures
            strategy: Current search strategy from curate phase
            reflection: Performance reflection from last iteration
            n_structures: Number of structures to generate

        Returns:
            List of tool actions:
            [
                {"tool": "substitute", "parent": 0, "old_element": "O", "new_element": "S"},
                {"tool": "mutate", "parent": 1, "strength": 0.1},
                {"tool": "new"}
            ]
        """
        logger.info("Tool selector analyzing state and choosing tools...")

        # Handle empty pool case
        if len(example_pool) == 0:
            logger.info("Empty parent pool - using only 'new' tool")
            return [{"tool": "new"} for _ in range(n_structures)]

        # Build decision prompt
        prompt = self._build_decision_prompt(
            example_pool, strategy, reflection, n_structures
        )

        # Get LLM decision
        try:
            response = self.llm.generate(prompt, n=1)[0]
            logger.debug(f"LLM tool selection response:\n{response}")

            # Parse response
            actions = self._parse_tool_actions(response, len(example_pool))

            logger.info(f"Selected {len(actions)} tool actions:")
            for action in actions:
                logger.info(f"  - {action['tool']}: {action}")

            return actions

        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            # Fallback: half new, half substitute
            logger.warning("Using fallback tool selection")
            return self._fallback_selection(example_pool, n_structures)

    def _build_decision_prompt(
        self,
        parents: List[CrystalStructure],
        strategy: Optional[str],
        reflection: Dict,
        n: int
    ) -> str:
        """Build prompt for tool selection"""

        prompt_parts = [
            "You are a materials scientist deciding how to generate new crystal structures.",
            "",
            "="*60,
            "AVAILABLE TOOLS:",
            "="*60,
            "",
            "1. substitute(parent_idx, old_element, new_element)",
            "   - Replace an element in a parent structure",
            "   - Example: substitute(0, 'O', 'S') changes ZnO to ZnS",
            "   - Use when: exploring chemical variations",
            "",
            "2. mutate(parent_idx, strength)",
            "   - Randomly perturb structure (strength: 0.05-0.2)",
            "   - Example: mutate(0, 0.1) makes small changes",
            "   - Use when: fine-tuning existing structures",
            "",
            "3. mix(parent1_idx, parent2_idx, ratio)",
            "   - Mix two parent structures (ratio: 0-1)",
            "   - Example: mix(0, 1, 0.6) combines 60% parent0 + 40% parent1",
            "   - Use when: combining advantages of two structures",
            "",
            "4. new()",
            "   - Generate entirely new structure from scratch",
            "   - Use when: exploring new chemistries",
            "",
            "="*60,
            "CURRENT SITUATION:",
            "="*60,
            ""
        ]

        # Add strategy
        if strategy:
            prompt_parts.extend([
                "STRATEGY:",
                strategy,
                ""
            ])

        # Add reflection
        if reflection:
            prompt_parts.extend([
                "PREVIOUS ITERATION RESULTS:",
                f"  - Best decomposition energy: {reflection.get('best_ed', 'N/A')} eV/atom",
                f"  - Valid rate: {reflection.get('valid_rate', 0):.1f}%",
                f"  - Diversity: {reflection.get('diversity', 0):.2f}",
                ""
            ])

        # Add parent structures
        prompt_parts.extend([
            "AVAILABLE PARENT STRUCTURES:",
            ""
        ])

        for i, parent in enumerate(parents[:5]):  # Show up to 5 parents
            ed = parent.decomposition_energy if parent.decomposition_energy else "N/A"
            prompt_parts.append(f"  Parent {i}: {parent.formula} (Ed={ed} eV/atom)")

        prompt_parts.extend([
            "",
            "="*60,
            "YOUR TASK:",
            "="*60,
            "",
            f"Decide how to generate {n} new structures.",
            "Consider the strategy and current results to choose appropriate tools.",
            "",
            "Output format (one action per line):",
            "",
            "substitute(parent=0, old='O', new='S')",
            "mutate(parent=1, strength=0.1)",
            "mix(parent1=0, parent2=1, ratio=0.6)",
            "new()",
            "",
            f"Provide exactly {n} actions now:"
        ])

        return "\n".join(prompt_parts)

    def _parse_tool_actions(self, response: str, num_parents: int) -> List[Dict]:
        """
        Parse LLM response into tool actions

        Extracts tool calls from response text
        """
        actions = []

        # Pattern matching for each tool type
        patterns = {
            'substitute': r"substitute\(parent=(\d+),\s*old='(\w+)',\s*new='(\w+)'\)",
            'mutate': r"mutate\(parent=(\d+),\s*strength=([\d.]+)\)",
            'mix': r"mix\(parent1=(\d+),\s*parent2=(\d+),\s*ratio=([\d.]+)\)",
            'new': r"new\(\)"
        }

        for line in response.split('\n'):
            line = line.strip()

            # Try each pattern
            if 'substitute' in line.lower():
                match = re.search(patterns['substitute'], line)
                if match:
                    parent_idx = int(match.group(1))
                    if parent_idx < num_parents:
                        actions.append({
                            "tool": "substitute",
                            "parent": parent_idx,
                            "old_element": match.group(2),
                            "new_element": match.group(3)
                        })

            elif 'mutate' in line.lower():
                match = re.search(patterns['mutate'], line)
                if match:
                    parent_idx = int(match.group(1))
                    if parent_idx < num_parents:
                        actions.append({
                            "tool": "mutate",
                            "parent": parent_idx,
                            "strength": float(match.group(2))
                        })

            elif 'mix' in line.lower():
                match = re.search(patterns['mix'], line)
                if match:
                    p1 = int(match.group(1))
                    p2 = int(match.group(2))
                    if p1 < num_parents and p2 < num_parents:
                        actions.append({
                            "tool": "mix",
                            "parent1": p1,
                            "parent2": p2,
                            "ratio": float(match.group(3))
                        })

            elif 'new()' in line.lower():
                actions.append({"tool": "new"})

        return actions

    def _fallback_selection(self, parents: List, n: int) -> List[Dict]:
        """Fallback tool selection if LLM fails"""
        actions = []

        if len(parents) == 0:
            return [{"tool": "new"} for _ in range(n)]

        # Simple heuristic: 50% substitute, 50% new
        for i in range(n):
            if i % 2 == 0 and len(parents) > 0:
                actions.append({
                    "tool": "substitute",
                    "parent": 0,
                    "old_element": "O",
                    "new_element": "S"
                })
            else:
                actions.append({"tool": "new"})

        return actions
