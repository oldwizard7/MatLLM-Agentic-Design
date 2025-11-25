"""Material database for filtering known materials with MP API integration"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# 基础已知材料列表（本地缓存，无需 API 调用）
KNOWN_MATERIALS = {
    # 电池材料
    'LiCoO2', 'LiFePO4', 'LiMn2O4', 'NaCoO2', 'LiNiO2',
    'LiMnO2', 'NaFePO4',
    
    # 钙钛矿
    'CaTiO3', 'BaTiO3', 'SrTiO3', 'BaZrO3', 'ZnTiO3',
    'PbTiO3', 'LaAlO3', 'SrRuO3',
    
    # 常见氧化物
    'MgO', 'Al2O3', 'SiO2', 'TiO2', 'Fe2O3', 'Fe3O4',
    'ZnO', 'CuO', 'Cu2O', 'SnO2', 'In2O3',
    
    # 硅酸盐/矿物
    'Mg2SiO4', 'MgSiO3', 'CaSiO3', 'Ca2SiO4',
    'Al2SiO5', 'Mg3Al2Si3O12',
    
    # 半导体
    'GaN', 'AlN', 'InN', 'GaAs', 'InP', 'GaP',
    'ZnS', 'ZnSe', 'CdS', 'CdSe', 'CdTe',
    'Si', 'Ge', 'SiC',
    
    # 卤化物
    'NaCl', 'KCl', 'LiF', 'NaF', 'MgF2', 'CaF2',
    'LiCl', 'KBr', 'NaBr',
    
    # 硫化物/硒化物
    'FeS2', 'MoS2', 'WS2', 'Bi2S3', 'Sb2S3',
    'PbS', 'Ag2S', 'Cu2S',
    
    # 氮化物
    'BN', 'Si3N4', 'AlN', 'TiN',
}


def normalize_formula(formula: str) -> str:
    """
    规范化化学式，用于比较
    
    例如:
        "LiCoO2 - Layered Structure" → "LiCoO2"
        "Li1Co1O2" → "LiCoO2"
        "Mg 2 Si O 4" → "Mg2SiO4"
    
    Args:
        formula: 原始化学式
        
    Returns:
        规范化后的化学式
    """
    import re
    
    # 1. 移除结构描述（" - " 后面的部分）
    if ' - ' in formula:
        formula = formula.split(' - ')[0].strip()
    
    # 2. 移除所有空格
    formula = re.sub(r'\s+', '', formula)
    
    # 3. 移除系数为1的数字（可选）
    # formula = re.sub(r'([A-Z][a-z]?)1(?![0-9])', r'\1', formula)
    
    return formula


def check_materials_project(formula: str, api_key: Optional[str] = None) -> bool:
    """
    使用 Materials Project API 检查材料是否存在
    
    Args:
        formula: 化学式
        api_key: MP API key (如果为 None，从环境变量读取)
        
    Returns:
        True = 在 MP 数据库中找到，False = 未找到或查询失败
    """
    try:
        from mp_api.client import MPRester
        
        # 获取 API key
        if api_key is None:
            api_key = os.getenv('MP_API_KEY')
        
        if not api_key:
            logger.warning("MP_API_KEY not found, skipping MP query")
            return False
        
        # 查询 Materials Project
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty"]
            )
            
            if docs:
                logger.debug(f"Found {len(docs)} entries in MP for {formula}")
                return True
            else:
                logger.debug(f"No MP entries found for {formula}")
                return False
                
    except ImportError:
        logger.warning("mp-api not installed. Install with: pip install mp-api")
        return False
    except Exception as e:
        logger.warning(f"MP API query failed for {formula}: {e}")
        return False


def is_known_material(
    formula: str, 
    use_mp: bool = True,
    strict: bool = False
) -> bool:
    """
    检查材料是否为已知材料（本地 + MP API）
    
    Args:
        formula: 化学式（可能包含描述，如 "LiCoO2 - Layered"）
        use_mp: 是否使用 Materials Project API（默认 True）
        strict: 严格模式（True）或宽松模式（False）
        
    Returns:
        True = 已知材料，False = 可能是新材料
    """
    # 规范化化学式
    clean_formula = normalize_formula(formula)
    
    # 1. 先检查本地数据库（快速）
    if clean_formula in KNOWN_MATERIALS:
        logger.debug(f"{formula} found in local database")
        return True
    
    # 2. 如果本地没有，查询 Materials Project（需要网络）
    if use_mp:
        if check_materials_project(clean_formula):
            # 找到了，添加到本地缓存
            KNOWN_MATERIALS.add(clean_formula)
            return True
    
    # 3. 本地和 MP 都没找到，认为是新材料
    return False


def categorize_materials(
    structures: list,
    use_mp: bool = True,
    progress_callback: Optional[callable] = None
) -> dict:
    """
    将结构列表分类为已知和新材料
    
    Args:
        structures: CrystalStructure 对象列表
        use_mp: 是否使用 Materials Project API
        progress_callback: 进度回调函数 callback(current, total)
        
    Returns:
        {
            'known': [已知材料列表],
            'novel': [新材料列表],
            'all': [所有材料列表]
        }
    """
    known = []
    novel = []
    
    total = len(structures)
    
    for i, struct in enumerate(structures):
        # 进度回调
        if progress_callback:
            progress_callback(i + 1, total)
        
        if is_known_material(struct.formula, use_mp=use_mp):
            known.append(struct)
        else:
            novel.append(struct)
    
    logger.info(f"Categorized {total} structures: {len(known)} known, {len(novel)} novel")
    
    return {
        'known': known,
        'novel': novel,
        'all': structures
    }


def add_known_material(formula: str):
    """
    手动添加一个已知材料到本地数据库
    
    Args:
        formula: 化学式
    """
    clean_formula = normalize_formula(formula)
    KNOWN_MATERIALS.add(clean_formula)
    logger.info(f"Added {clean_formula} to known materials database")


def get_statistics(categorized: dict) -> dict:
    """
    获取分类后的统计信息
    
    Args:
        categorized: categorize_materials() 的返回结果
        
    Returns:
        统计字典
    """
    all_structures = categorized['all']
    known = categorized['known']
    novel = categorized['novel']
    
    # 稳定性统计
    novel_stable = [s for s in novel if s.is_valid and s.decomposition_energy < 0]
    known_stable = [s for s in known if s.is_valid and s.decomposition_energy < 0]
    
    stats = {
        'total': len(all_structures),
        'known_count': len(known),
        'novel_count': len(novel),
        'known_ratio': len(known) / len(all_structures) if all_structures else 0,
        'novel_ratio': len(novel) / len(all_structures) if all_structures else 0,
        'novel_stable_count': len(novel_stable),
        'known_stable_count': len(known_stable),
    }
    
    # 最佳新材料
    if novel_stable:
        best_novel = min(novel_stable, key=lambda s: s.decomposition_energy)
        stats['best_novel'] = {
            'formula': best_novel.formula,
            'decomposition_energy': best_novel.decomposition_energy
        }
    
    return stats


# 测试函数
if __name__ == "__main__":
    import sys
    
    # 测试用例
    test_cases = [
        "LiCoO2 - Layered Structure",
        "BaZrO3 - Perovskite Structure",
        "BeNO2 - Nitridoberyllate Structure",
        "AlPSe2",
        "MgSiO4 - Disilicate Structure",
        "CuGaN2",  # 假设的新材料
        "Sc2O3",   # 氧化钪，应该在 MP 中
    ]
    
    print("Material Classification Test")
    print("=" * 70)
    print(f"{'Formula':<40} {'Local':<10} {'MP API':<10} {'Status':<10}")
    print("=" * 70)
    
    for formula in test_cases:
        clean = normalize_formula(formula)
        in_local = clean in KNOWN_MATERIALS
        in_mp = check_materials_project(clean)
        is_known = is_known_material(formula, use_mp=True)
        
        status = "KNOWN" if is_known else "NOVEL"
        local_str = "✓" if in_local else "✗"
        mp_str = "✓" if in_mp else "✗"
        
        print(f"{formula:<40} {local_str:<10} {mp_str:<10} {status:<10}")
