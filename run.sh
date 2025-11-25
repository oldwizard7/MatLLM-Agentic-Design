#!/bin/bash
# MatAgent 运行脚本
# 使用方法: ./run.sh [--quick] [--no-mp]
#   --quick: 快速测试模式（2次迭代）
#   --no-mp: 不使用 Materials Project API

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 解析参数
QUICK_MODE=false
USE_MP=true

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --no-mp)
            USE_MP=false
            shift
            ;;
        *)
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MatAgent - Crystal Structure Discovery${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 检查环境
echo -e "${GREEN}[1/6] Checking environment...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"

# 检查 conda 环境
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "  Conda env: $CONDA_DEFAULT_ENV"
else
    echo -e "${YELLOW}  Warning: Not in a conda environment${NC}"
fi

# 2. 加载 .env 文件（如果存在）
if [ -f ".env" ]; then
    echo -e "${GREEN}Loading .env file...${NC}"
    export $(cat .env | grep -v '^#' | xargs)
    echo "  .env file loaded ✓"
fi

# 3. 检查 API keys
echo -e "${GREEN}[2/6] Checking API keys...${NC}"
if [ -z "$OPENAI_API_KEY" ] || [[ "$OPENAI_API_KEY" == "your_"* ]]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set or invalid${NC}"
    echo "Please edit .env file and add your OpenAI API key:"
    echo "  nano .env"
    exit 1
else
    echo "  OpenAI: ✓ (${OPENAI_API_KEY:0:10}...)"
fi

if $USE_MP; then
    if [ -z "$MP_API_KEY" ] || [[ "$MP_API_KEY" == "your_"* ]]; then
        echo -e "${YELLOW}  Warning: MP_API_KEY not set or invalid${NC}"
        echo "  Materials Project filtering will be disabled"
        USE_MP=false
    else
        echo "  MP API: ✓ (${MP_API_KEY:0:10}...)"
    fi
else
    echo "  MP API: Disabled (--no-mp flag)"
fi

# 3. 检查依赖
echo -e "${GREEN}[3/6] Checking dependencies...${NC}"
python -c "import chgnet" 2>/dev/null && echo "  CHGNet: ✓" || echo -e "${YELLOW}  CHGNet: ✗ (will be loaded on demand)${NC}"
python -c "import openai" 2>/dev/null && echo "  OpenAI: ✓" || { echo -e "${RED}  OpenAI: ✗${NC}"; exit 1; }

if $USE_MP; then
    python -c "import mp_api" 2>/dev/null && echo "  MP-API: ✓" || echo -e "${YELLOW}  MP-API: ✗ (install: pip install mp-api)${NC}"
fi

# 4. 创建输出目录
echo -e "${GREEN}[4/6] Creating output directory...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if $QUICK_MODE; then
    OUTPUT_DIR="data/results/quick_test_${TIMESTAMP}"
    echo "  Mode: Quick test (2 iterations)"
else
    OUTPUT_DIR="data/results/run_${TIMESTAMP}"
    echo "  Mode: Full run (10 iterations)"
fi
mkdir -p "$OUTPUT_DIR"
echo "  Output: $OUTPUT_DIR"

# 5. 备份配置（如果是快速模式，临时修改）
echo -e "${GREEN}[5/6] Configuring...${NC}"
if $QUICK_MODE; then
    cp config/config.yaml config/config.yaml.backup
    # 临时修改为快速模式
    sed -i 's/max_iterations: 10/max_iterations: 2/' config/config.yaml
    sed -i 's/children_size: 10/children_size: 3/' config/config.yaml
    echo "  Quick mode: 2 iterations, 3 children per iteration"
fi

# 6. 运行主程序
echo -e "${GREEN}[6/6] Running MatAgent...${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

python src/main.py \
  --data data/reference_examples/ \
  --output "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/run.log"

EXIT_CODE=$?

# 恢复配置
if $QUICK_MODE; then
    mv config/config.yaml.backup config/config.yaml
fi

# 7. 总结
echo ""
echo -e "${BLUE}========================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Complete!${NC}"
else
    echo -e "${RED}❌ Failed with exit code $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"

# 统计文件
JSON_COUNT=$(ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)
PNG_COUNT=$(ls "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)

if [ -f "$OUTPUT_DIR/novel_0.json" ]; then
    NOVEL_COUNT=$(ls "$OUTPUT_DIR"/novel_*.json 2>/dev/null | wc -l)
    KNOWN_COUNT=$(ls "$OUTPUT_DIR"/known_*.json 2>/dev/null | wc -l)
    echo "  Novel structures: $NOVEL_COUNT"
    echo "  Known structures: $KNOWN_COUNT"
fi

echo "  Total JSON files: $JSON_COUNT"
echo "  Plots: $PNG_COUNT"
echo ""
echo "View results:"
echo "  cd $OUTPUT_DIR"
echo "  ls -lh"
echo ""
echo -e "${BLUE}========================================${NC}"