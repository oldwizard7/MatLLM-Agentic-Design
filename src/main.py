"""Main entry point for MatLLMSearch"""

import logging
import yaml
import argparse
from pathlib import Path
import numpy as np
import sys

# Ensure the project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from src.utils.env_loader import load_env_file, check_environment
load_env_file()

from src.agents.orchestrator import OrchestratorAgent
from src.utils.visualization import ResultsVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prompts(prompts_path: str) -> dict:
    """Load prompt templates from YAML"""
    with open(prompts_path, 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts


def save_results(structures: list, output_dir: str, name: str):
    """Save structures to output directory"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, struct in enumerate(structures):
        struct.save(str(output_path / f"{name}_{i}.json"))


def main():
    """Main execution loop"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='MatLLMSearch - LLM-assisted crystal structure discovery')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--prompts', type=str, default='config/prompts.yaml',
                       help='Path to prompts file')
    ####
    #这个data里的initial_structures没搞定,可以先用example
    parser.add_argument('--data', type=str, default='data/reference_examples/',
                   help='Path to reference examples (format demonstration)')
    parser.add_argument('--output', type=str, default='data/results/',
                       help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    prompts = load_prompts(args.prompts)
    
    # Initialize orchestrator
    logger.info("Initializing Orchestrator Agent...")
    orchestrator = OrchestratorAgent(config, prompts)
    
    # Initialize parent pool
    logger.info(f"Loading initial structures from {args.data}")
    filters = {
        'num_elements': (3, 6),  # 3-6 elements
        'exclude_elements': []    # Exclude f-block elements if desired
    }
    # Initialize with reference examples (not parent pool)
    example_pool = orchestrator.initialize(args.data, filters)
    
    # example_pool可以为空
    # if not example_pool:
    #     logger.error("No initial structures loaded. Exiting.")
    #     return
    
    # ACE Main Loop
    logger.info("\n" + "="*60)
    logger.info("Starting MatLLMSearch ACE Loop")
    logger.info("="*60 + "\n")
    
    all_structures = []
    
    while True:
        orchestrator.iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {orchestrator.iteration}")
        logger.info(f"{'='*60}\n")
        
        # 1. GENERATE: Create new structures
        children = orchestrator.generate(example_pool)
        
        if not children:
            logger.warning("No children generated. Terminating.")
            break
        
        # 2. REFLECT: Analyze results
        reflection = orchestrator.reflect(children)

        # 3. CURATE: Evolve strategy (pass example_pool for analysis)
        new_plan = orchestrator.curate(reflection, example_pool)

        # 4. SELECT: Form next generation
        from src.core.evolution import EvolutionEngine
        evolution = EvolutionEngine(config)
        example_pool = evolution.select(example_pool, children)
        
        # Store all structures
        all_structures.extend(children)
        
        # Save checkpoint
        if orchestrator.iteration % config['logging']['save_frequency'] == 0:
            orchestrator.save_checkpoint(example_pool, args.output)
        
        # Check termination
        if orchestrator.should_terminate(reflection):
            logger.info("\nTermination condition met.")
            break
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("SEARCH COMPLETE")
    logger.info("="*60)
    
    # Save all structures
    # ========== 过滤已知材料 (使用 MP API) ==========
    logger.info("\n" + "="*60)
    logger.info("FILTERING KNOWN MATERIALS")
    logger.info("="*60)
    
    from src.utils.material_database import categorize_materials, get_statistics

    # 进度回调
    def progress_callback(current, total):
        if current % 5 == 0 or current == total:
            logger.info(f"  Checking materials: {current}/{total}")
    
    # 分类材料（使用 MP API）
    logger.info("Querying Materials Project database...")
    categorized = categorize_materials(
        all_structures,
        use_mp=True,  # 启用 MP API
        progress_callback=progress_callback
    )
    
    known_structures = categorized['known']
    novel_structures = categorized['novel']
    
    # 获取统计信息
    stats = get_statistics(categorized)
    
    # 输出分类结果
    logger.info(f"\n{'='*60}")
    logger.info("MATERIAL CATEGORIZATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total structures: {stats['total']}")
    logger.info(f"Known materials: {stats['known_count']} ({100*stats['known_ratio']:.1f}%)")
    logger.info(f"Novel materials: {stats['novel_count']} ({100*stats['novel_ratio']:.1f}%)")
    
    if known_structures:
        logger.info(f"\n{'Known materials found':}")
        for s in sorted(known_structures, key=lambda x: x.decomposition_energy or 0):
            ed = s.decomposition_energy if s.decomposition_energy else float('inf')
            logger.info(f"  - {s.formula}: Ed = {ed:.3f} eV/atom")
    
    if novel_structures:
        logger.info(f"\n{'Novel materials found':}")
        for s in sorted(novel_structures, key=lambda x: x.decomposition_energy or 0):
            ed = s.decomposition_energy if s.decomposition_energy else float('inf')
            logger.info(f"  - {s.formula}: Ed = {ed:.3f} eV/atom")
    
    # ========== 保存不同类别的结果 ==========
    logger.info(f"\n{'='*60}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*60}")
    
    # 保存新材料（重点）
    if novel_structures:
        logger.info(f"Saving {len(novel_structures)} novel structures...")
        save_results(novel_structures, args.output, "novel")
    
    # 保存已知材料（用于分析）
    if known_structures:
        logger.info(f"Saving {len(known_structures)} known structures...")
        save_results(known_structures, args.output, "known")
    
    # 保存全部（备份）
    logger.info(f"Saving all {len(all_structures)} structures...")
    save_results(all_structures, args.output, "all")
    
    
    
    logger.info(f"\nSaving {len(all_structures)} structures to {args.output}")
    save_results(all_structures, args.output, "final")
    
    # Generate visualizations
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*60)
    
    visualizer = ResultsVisualizer()
    
    # 为新材料生成可视化
    if novel_structures:
        logger.info("Creating energy distribution for novel materials...")
        visualizer.plot_energy_distribution(
            novel_structures,
            save_path=f"{args.output}/energy_distribution_novel.png"
        )
    
    # 为全部材料也生成一份（对比用）
    logger.info("Creating energy distribution for all materials...")
    visualizer.plot_energy_distribution(
        all_structures,
        save_path=f"{args.output}/energy_distribution_all.png"
    )
    
    logger.info("Creating convergence plot...")
    visualizer.plot_convergence(
        orchestrator.search_history,
        save_path=f"{args.output}/convergence.png"
    )
    
    # Summary statistics 
    logger.info("\n" + "="*60)
    logger.info("FINAL STATISTICS - ALL STRUCTURES")
    logger.info("="*60)
    
    valid = [s for s in all_structures if s.is_valid]
    metastable = [s for s in valid if s.decomposition_energy < 0.1]
    stable = [s for s in metastable if s.decomposition_energy < 0]
    
    logger.info(f"Total generated: {len(all_structures)}")
    
    if len(all_structures) > 0:
        logger.info(f"Valid: {len(valid)} ({100*len(valid)/len(all_structures):.1f}%)")
        logger.info(f"Metastable: {len(metastable)} ({100*len(metastable)/len(all_structures):.1f}%)")
        logger.info(f"Stable: {len(stable)} ({100*len(stable)/len(all_structures):.1f}%)")
    
    # 新材料统计
    logger.info("\n" + "="*60)
    logger.info("FINAL STATISTICS - NOVEL MATERIALS ONLY")
    logger.info("="*60)
    
    if novel_structures:
        novel_valid = [s for s in novel_structures if s.is_valid]
        novel_stable = [s for s in novel_valid if s.decomposition_energy < 0]
        
        logger.info(f"Novel structures: {len(novel_structures)}")
        logger.info(f"Novel valid: {len(novel_valid)} ({100*len(novel_valid)/len(novel_structures):.1f}%)")
        logger.info(f"Novel stable: {len(novel_stable)} ({100*len(novel_stable)/len(novel_structures):.1f}%)")
        
        if novel_stable:
            best_novel = min(novel_stable, key=lambda s: s.decomposition_energy)
            logger.info(f"\n🌟 BEST NOVEL STRUCTURE:")
            logger.info(f"   Formula: {best_novel.formula}")
            logger.info(f"   Decomposition energy: {best_novel.decomposition_energy:.3f} eV/atom")
            logger.info(f"   This material is NOT in Materials Project database!")
    else:
        logger.warning("⚠️  No novel materials discovered in this run!")
        logger.info("All generated structures are known materials.")
        logger.info("Consider adjusting the search strategy or prompt.")
    
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"  - novel_*.json: {len(novel_structures)} novel materials")
    logger.info(f"  - known_*.json: {len(known_structures)} known materials")
    logger.info(f"  - all_*.json: {len(all_structures)} total materials")
    logger.info(f"  - *.png: Visualization plots")


    
    # Summary statistics 统计输出部分
    
    logger.info("\n" + "="*60)
    logger.info("FINAL STATISTICS")
    logger.info("="*60)
    
    valid = [s for s in all_structures if s.is_valid]
    metastable = [s for s in valid if s.decomposition_energy < 0.1]
    stable = [s for s in metastable if s.decomposition_energy < 0]
    
    logger.info(f"Total generated: {len(all_structures)}")
    
    
    if len(all_structures) > 0:
        logger.info(f"Valid: {len(valid)} ({100*len(valid)/len(all_structures):.1f}%)")
        logger.info(f"Metastable: {len(metastable)} ({100*len(metastable)/len(all_structures):.1f}%)")
        logger.info(f"Stable: {len(stable)} ({100*len(stable)/len(all_structures):.1f}%)")
        
        if stable:
            best = min(stable, key=lambda s: s.decomposition_energy)
            logger.info(f"\nBest structure: {best.formula}")
            logger.info(f"Decomposition energy: {best.decomposition_energy:.3f} eV/atom")
    else:
        logger.warning("No structures were generated in this run.")
    
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
