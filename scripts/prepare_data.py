"""Prepare data for MatAgent"""

import sys
sys.path.insert(0, '.')

import argparse
import logging
import os
from src.utils.data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Prepare data for MatLLMSearch')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['download', 'examples', 'cache'],
                       help='Data preparation mode')
    
    parser.add_argument('--api-key', type=str,
                       help='Materials Project API key (for download/cache)')
    
    parser.add_argument('--max-structures', type=int, default=1000,
                       help='Maximum structures to download')
    
    parser.add_argument('--elements', type=str, nargs='+',
                       help='Include only these elements')
    
    parser.add_argument('--exclude-elements', type=str, nargs='+',
                       help='Exclude these elements (e.g., f-block)')
    
    args = parser.parse_args()
    
    # Initialize data manager
    dm = DataManager()
    
    #这个是正经去下载的
    if args.mode == 'download':
        # Download from Materials Project
        if not args.api_key:
            api_key = os.getenv('MP_API_KEY')
            if not api_key:
                logger.error("Materials Project API key required!")
                logger.error("Set MP_API_KEY environment variable or use --api-key")
                return
        else:
            api_key = args.api_key
        
        filters = {}
        if args.elements:
            filters['elements'] = args.elements
        if args.exclude_elements:
            filters['exclude_elements'] = args.exclude_elements
        
        logger.info(f"Downloading up to {args.max_structures} structures...")
        structures = dm.download_from_materials_project(
            api_key=api_key,
            max_structures=args.max_structures,
            filters=filters
        )
        
        logger.info(f"✓ Downloaded {len(structures)} structures to data/initial_structures/")
    
    
    #纯供测试
    elif args.mode == 'examples':
        # Create example dataset
        logger.info("Creating example dataset...")
        dm.create_example_dataset()
        logger.info("✓ Created example structures in data/examples/")
    
    elif args.mode == 'cache':
        # Cache phase diagram
        if not args.api_key:
            api_key = os.getenv('MP_API_KEY')
            if not api_key:
                logger.error("Materials Project API key required!")
                return
        else:
            api_key = args.api_key
        
        logger.info("Caching phase diagram...")
        dm.cache_phase_diagram(api_key)
        logger.info("✓ Cached phase diagram to data/cache/")


if __name__ == "__main__":
    main()