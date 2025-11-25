#!/usr/bin/env python
"""Test ORB-v3 integration"""

import sys
import logging
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_orb_installation():
    """Test if ORB is installed"""
    print("="*60)
    print("Testing ORB Installation")
    print("="*60)

    try:
        import orb_models
        print(f"✅ orb-models version: {orb_models.__version__}")
        return True
    except ImportError as e:
        print(f"❌ orb-models not installed: {e}")
        print("\nInstall with:")
        print("  pip install orb-models")
        return False

def test_orb_loading():
    """Test loading ORB model"""
    print("\n" + "="*60)
    print("Testing ORB Model Loading")
    print("="*60)

    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        print("Loading ORB-v3 model (this may take a moment)...")
        # Use conservative model with Materials Project-like training data
        orbff = pretrained.orb_v3_conservative_20_mpa(device='cpu')
        calculator = ORBCalculator(orbff, device='cpu')

        print("✅ ORB-v3 model loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to load ORB model: {e}")
        return False

def test_evaluator():
    """Test StructureEvaluator with ORB backend"""
    print("\n" + "="*60)
    print("Testing StructureEvaluator with ORB")
    print("="*60)

    try:
        from src.core.structure import CrystalStructure
        from src.core.evaluator import StructureEvaluator
        import numpy as np

        # Create test structure (NaCl)
        struct = CrystalStructure(
            formula='NaCl',
            lattice=np.eye(3) * 5.64,
            positions=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
            species=['Na', 'Cl']
        )

        # Create evaluator with ORB backend
        config = {
            'evaluator': {
                'backend': 'orb',
                'model_name': 'orb-v3',
                'device': 'cpu',
                'relax': {
                    'fmax': 0.1,
                    'steps': 10,  # Only 10 steps for quick test
                    'optimizer': 'FIRE'
                }
            }
        }

        print("Creating evaluator...")
        evaluator = StructureEvaluator(config)

        print(f"Testing relaxation on {struct.formula}...")
        relaxed = evaluator.relax_structure(struct)

        if relaxed.is_valid:
            print(f"✅ Relaxation successful!")
            print(f"   Formula: {relaxed.formula}")
            print(f"   Relaxation steps: {relaxed.metadata.get('relaxation_steps', 'N/A')}")
            print(f"   Backend: {relaxed.metadata.get('backend', 'N/A')}")

            # Test energy calculation
            energy = evaluator.calculate_energy(relaxed)
            print(f"   Energy: {energy:.4f} eV/atom")

            return True
        else:
            print("❌ Relaxation failed")
            return False

    except Exception as e:
        print(f"❌ Evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_backends():
    """Compare CHGNet vs ORB on same structure"""
    print("\n" + "="*60)
    print("Comparing CHGNet vs ORB-v3")
    print("="*60)

    try:
        from src.core.structure import CrystalStructure
        from src.core.evaluator import StructureEvaluator
        import numpy as np

        # Create test structure
        struct = CrystalStructure(
            formula='LiCoO2',
            lattice=np.array([
                [2.8, 0.0, 0.0],
                [0.0, 2.8, 0.0],
                [0.0, 0.0, 14.0]
            ]),
            positions=np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.0, 0.5, 0.75],
                [0.5, 0.0, 0.75]
            ]),
            species=['Li', 'Co', 'O', 'O']
        )

        results = {}

        for backend in ['chgnet', 'orb']:
            print(f"\nTesting {backend.upper()}...")

            config = {
                'evaluator': {
                    'backend': backend,
                    'model_name': 'orb-v3' if backend == 'orb' else None,
                    'device': 'cpu',
                    'relax': {
                        'fmax': 0.1,
                        'steps': 10,
                        'optimizer': 'FIRE'
                    }
                }
            }

            try:
                evaluator = StructureEvaluator(config)
                # Create fresh structure for each backend test
                test_struct = CrystalStructure(
                    formula='LiCoO2',
                    lattice=np.array([
                        [2.8, 0.0, 0.0],
                        [0.0, 2.8, 0.0],
                        [0.0, 0.0, 14.0]
                    ]),
                    positions=np.array([
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5],
                        [0.0, 0.5, 0.75],
                        [0.5, 0.0, 0.75]
                    ]),
                    species=['Li', 'Co', 'O', 'O']
                )
                relaxed = evaluator.relax_structure(test_struct)

                if relaxed.is_valid:
                    energy = evaluator.calculate_energy(relaxed)
                    results[backend] = {
                        'energy': energy,
                        'steps': relaxed.metadata.get('relaxation_steps', 0)
                    }
                    print(f"  ✅ Energy: {energy:.4f} eV/atom")
                    print(f"     Steps: {results[backend]['steps']}")
                else:
                    print(f"  ❌ Failed")

            except Exception as e:
                print(f"  ❌ Error: {e}")

        # Compare results
        if len(results) == 2:
            print("\n" + "-"*60)
            print("Comparison:")
            print("-"*60)
            chg_e = results['chgnet']['energy']
            orb_e = results['orb']['energy']
            diff = abs(chg_e - orb_e)

            print(f"CHGNet:  {chg_e:.4f} eV/atom")
            print(f"ORB-v3:  {orb_e:.4f} eV/atom")
            print(f"Diff:    {diff:.4f} eV/atom ({100*diff/abs(chg_e):.2f}%)")

        return True

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "🚀 ORB-v3 Integration Test Suite")
    print("="*60)

    tests = [
        ("Installation", test_orb_installation),
        ("Model Loading", test_orb_loading),
        ("Evaluator", test_evaluator),
        ("Backend Comparison", compare_backends),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} | {name}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed!")
        print("ORB-v3 is ready to use.")
        print("\nTo use ORB-v3, set in config.yaml:")
        print("  evaluator:")
        print("    backend: orb")
        print("    model_name: orb-v3")
    else:
        print("⚠️  Some tests failed")
        print("Check the errors above and fix them.")

    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
