"""Structure evaluation using Machine Learning Interatomic Potentials - Multi-Backend Support"""

from typing import List, Optional, Dict
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.entries.computed_entries import ComputedEntry
from mp_api.client import MPRester
import ase
from ase.optimize import FIRE, BFGS, LBFGS
import logging

from .structure import CrystalStructure

logger = logging.getLogger(__name__)


class StructureEvaluator:
    """
    Evaluate crystal structures using Machine Learning Interatomic Potentials

    Supported backends:
    - CHGNet: Universal ML potential from Materials Project
    - M3GNet: Graph network potential
    - ORB: Orbital Materials foundation model (v1, v2, v3)

    Features:
    - Structure relaxation
    - Energy calculation
    - Decomposition energy (distance to convex hull)
    - Property calculation (bulk modulus, etc.)
    """

    def __init__(self, config: Dict):
        """
        Initialize evaluator

        Args:
            config: Configuration dictionary with 'evaluator' section
        """
        self.config = config
        eval_config = config.get('evaluator', {})

        # Get backend type
        self.backend = eval_config.get('backend', 'chgnet').lower()

        # Load appropriate model and calculator
        logger.info(f"Loading {self.backend.upper()} model...")
        self.model, self.calculator = self._load_model(eval_config)

        # Relaxation parameters
        relax_config = eval_config.get('relax', {})
        self.fmax = relax_config.get('fmax', 0.1)
        self.max_steps = relax_config.get('steps', 500)
        self.optimizer_name = relax_config.get('optimizer', 'FIRE')

        # Phase diagram for stability calculation
        self.phase_diagram = None
        self._load_phase_diagram()

    def _load_model(self, eval_config: Dict):
        """
        Load ML potential model based on backend

        Args:
            eval_config: Evaluator configuration

        Returns:
            (model, calculator) tuple
        """
        if self.backend == 'chgnet':
            try:
                from chgnet.model import CHGNet
                from chgnet.model.dynamics import CHGNetCalculator

                model = CHGNet.load()
                calculator = CHGNetCalculator(model)
                logger.info("CHGNet loaded successfully")
                return model, calculator

            except ImportError:
                logger.error("CHGNet not installed. Install with: pip install chgnet")
                raise

        elif self.backend == 'm3gnet':
            try:
                from matgl.ext.ase import M3GNetCalculator
                import matgl

                model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
                calculator = M3GNetCalculator(potential=model)
                logger.info("M3GNet loaded successfully")
                return model, calculator

            except ImportError:
                logger.error("M3GNet not installed. Install with: pip install matgl")
                raise

        elif self.backend == 'orb':
            try:
                from orb_models.forcefield import pretrained
                from orb_models.forcefield.calculator import ORBCalculator

                model_name = eval_config.get('model_name', 'orb-v3')
                device = eval_config.get('device', 'cpu')  # 'cpu' or 'cuda'

                # Load ORB model
                logger.info(f"Loading ORB model: {model_name} on {device}")

                if model_name == 'orb-v1' or model_name == 'orb_v1':
                    orbff = pretrained.orb_v1(device=device)
                elif model_name == 'orb-v2' or model_name == 'orb_v2':
                    orbff = pretrained.orb_v2(device=device)
                elif model_name in ['orb-v3', 'orb_v3', 'orb']:
                    # Use conservative model with 20Å cutoff trained on Materials Project-like data
                    # This is best for crystal structure evaluation
                    orbff = pretrained.orb_v3_conservative_20_mpa(device=device)
                elif model_name == 'orb_v3_direct':
                    orbff = pretrained.orb_v3_direct_20_mpa(device=device)
                elif hasattr(pretrained, model_name):
                    # Allow direct specification of any pretrained model
                    orbff = getattr(pretrained, model_name)(device=device)
                else:
                    raise ValueError(f"Unknown ORB model: {model_name}")

                calculator = ORBCalculator(orbff, device=device)
                logger.info(f"{model_name.upper()} loaded successfully on {device}")
                return orbff, calculator

            except ImportError as e:
                logger.error(f"ORB not installed: {e}")
                logger.error("Install with: pip install orb-models")
                raise

        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                f"Supported: chgnet, m3gnet, orb"
            )

    def _load_phase_diagram(self):
        """Load Materials Project phase diagram"""
        try:
            logger.info("Loading Materials Project phase diagram...")
            # This would load from MP API or cached data
            # For now, we'll compute it on demand
            pass
        except Exception as e:
            logger.warning(f"Could not load phase diagram: {e}")

    def evaluate(self, structure: CrystalStructure) -> CrystalStructure:
        """
        Complete evaluation pipeline

        Args:
            structure: Input structure

        Returns:
            Evaluated structure with computed properties
        """
        # 1. Relax structure
        relaxed = self.relax_structure(structure)

        if not relaxed.is_valid:
            # Relaxation failed, return invalid structure
            return relaxed

        # 2. Calculate energy
        relaxed.energy = self.calculate_energy(relaxed)

        # 3. Calculate decomposition energy
        relaxed.decomposition_energy = self.calculate_decomposition_energy(relaxed)

        # 4. Calculate additional properties
        if 'bulk_modulus' in self.config.get('properties', []):
            relaxed.properties['bulk_modulus'] = self.calculate_bulk_modulus(relaxed)

        relaxed.is_valid = True
        relaxed.is_relaxed = True

        return relaxed

    def relax_structure(self, structure: CrystalStructure) -> CrystalStructure:
        """
        Relax structure using the selected MLIP

        Args:
            structure: Input structure

        Returns:
            Relaxed structure
        """
        try:
            # Convert to ASE atoms
            pmg_struct = structure.to_pymatgen()
            atoms = ase.Atoms(
                symbols=[str(s) for s in pmg_struct.species],  # Convert Element objects to strings
                positions=pmg_struct.cart_coords,
                cell=pmg_struct.lattice.matrix,
                pbc=True
            )

            # Set calculator
            atoms.calc = self.calculator

            # Choose optimizer
            if self.optimizer_name.upper() == 'FIRE':
                optimizer = FIRE(atoms, logfile=None)
            elif self.optimizer_name.upper() == 'BFGS':
                optimizer = BFGS(atoms, logfile=None)
            elif self.optimizer_name.upper() == 'LBFGS':
                optimizer = LBFGS(atoms, logfile=None)
            else:
                logger.warning(f"Unknown optimizer {self.optimizer_name}, using FIRE")
                optimizer = FIRE(atoms, logfile=None)

            # Run relaxation
            optimizer.run(fmax=self.fmax, steps=self.max_steps)

            # Convert back to CrystalStructure
            relaxed_struct = Structure(
                lattice=atoms.cell[:],
                species=atoms.get_chemical_symbols(),
                coords=atoms.get_positions(),
                coords_are_cartesian=True
            )

            relaxed = CrystalStructure.from_pymatgen(relaxed_struct)
            relaxed.metadata['relaxation_steps'] = optimizer.get_number_of_steps()
            relaxed.metadata['backend'] = self.backend
            relaxed.is_valid = True
            relaxed.is_relaxed = True

            return relaxed

        except Exception as e:
            logger.error(f"Relaxation failed with {self.backend}: {e}")
            structure.is_valid = False
            return structure

    def calculate_energy(self, structure: CrystalStructure) -> float:
        """
        Calculate total energy

        Args:
            structure: Crystal structure

        Returns:
            Energy per atom (eV/atom)
        """
        try:
            pmg_struct = structure.to_pymatgen()
            
            #控制选择MILPs的种类
            if self.backend in ['chgnet', 'm3gnet']:
                # CHGNet and M3GNet use predict_structure
                prediction = self.model.predict_structure(pmg_struct)
                total_energy = prediction['e']  # eV
                energy_per_atom = total_energy / structure.num_atoms

            elif self.backend == 'orb':
                # ORB uses ASE calculator
                atoms = ase.Atoms(
                    symbols=[str(s) for s in pmg_struct.species],
                    positions=pmg_struct.cart_coords,
                    cell=pmg_struct.lattice.matrix,
                    pbc=True
                )
                atoms.calc = self.calculator
                total_energy = atoms.get_potential_energy()  # eV
                energy_per_atom = total_energy / len(atoms)

            else:
                raise ValueError(f"Energy calculation not implemented for {self.backend}")

            return energy_per_atom

        except Exception as e:
            logger.error(f"Energy calculation failed: {e}")
            return float('inf')

    def calculate_decomposition_energy(self, structure: CrystalStructure) -> float:
        """
        Calculate decomposition energy (distance to convex hull)

        Args:
            structure: Crystal structure

        Returns:
            Decomposition energy (eV/atom)
        """
        try:
            pmg_struct = structure.to_pymatgen()
            composition = pmg_struct.composition

            # Get competing phases from MP (cached)
            competing_energies = self._get_competing_phases(composition)

            if competing_energies is None:
                logger.warning("No phase diagram data, returning energy as Ed")
                return structure.energy

            # Calculate hull energy
            hull_energy = min(competing_energies)
            ed = structure.energy - hull_energy

            return ed

        except Exception as e:
            logger.error(f"Ed calculation failed: {e}")
            return float('inf')

    def _get_competing_phases(self, composition) -> Optional[List[float]]:
        """Get competing phase energies from Materials Project"""
        # This should query MP API or use cached data
        # Placeholder implementation
        return None

    def calculate_bulk_modulus(self, structure: CrystalStructure) -> float:
        """
        Calculate bulk modulus using Birch-Murnaghan equation of state

        Args:
            structure: Crystal structure

        Returns:
            Bulk modulus (GPa)
        """
        try:
            # Apply small strain and compute energy
            strains = np.linspace(-0.05, 0.05, 11)
            volumes = []
            energies = []

            for strain in strains:
                # Scale lattice
                scale = (1 + strain) ** (1/3)
                strained_lattice = structure.lattice * scale

                strained = CrystalStructure(
                    formula=structure.formula,
                    lattice=strained_lattice,
                    positions=structure.positions.copy(),
                    species=structure.species.copy()
                )

                energy = self.calculate_energy(strained)
                volume = strained.volume

                volumes.append(volume)
                energies.append(energy * structure.num_atoms)  # Total energy

            # Fit Birch-Murnaghan EOS
            from scipy.optimize import curve_fit

            def birch_murnaghan(V, E0, V0, B0, B0_prime):
                eta = (V0 / V) ** (2/3)
                return E0 + (9 * V0 * B0 / 16) * ((eta - 1) ** 2) * (
                    6 + B0_prime * (eta - 1) - 4 * eta
                )

            # Initial guess
            V0_guess = structure.volume
            E0_guess = min(energies)
            B0_guess = 100  # GPa

            popt, _ = curve_fit(
                birch_murnaghan,
                volumes,
                energies,
                p0=[E0_guess, V0_guess, B0_guess, 4]
            )

            bulk_modulus = popt[2]  # GPa

            return bulk_modulus

        except Exception as e:
            logger.error(f"Bulk modulus calculation failed: {e}")
            return 0.0

    def batch_evaluate(self, structures: List[CrystalStructure]) -> List[CrystalStructure]:
        """
        Evaluate multiple structures in batch

        Args:
            structures: List of structures

        Returns:
            List of evaluated structures
        """
        evaluated = []
        for i, struct in enumerate(structures):
            logger.info(f"Evaluating structure {i+1}/{len(structures)}: {struct.formula}")
            evaluated_struct = self.evaluate(struct)
            evaluated.append(evaluated_struct)

        return evaluated
