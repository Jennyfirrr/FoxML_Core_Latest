#!/bin/bash
# Build FoxML documentation as a single PDF using Pandoc + XeLaTeX.
#
# Usage:
#   bin/build-book-pdf.sh
#
# Output: book/FoxML_Documentation.pdf
#
# Dependencies (system packages):
#   pandoc, texlive-xetex (or equivalent)
#
# The file list below follows the same reading order as the HTML book nav.
# Only key documents are included (not every changelog entry).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Check dependencies ───────────────────────────────────────────
for cmd in pandoc xelatex; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: '$cmd' not found. Install pandoc and texlive-xetex."
        exit 1
    fi
done

# ── Create symlinks (same as HTML build) ─────────────────────────
ln -sfn "$REPO_ROOT/.claude/skills" "$REPO_ROOT/claude-skills"
ln -sfn "$REPO_ROOT/.claude/plans"  "$REPO_ROOT/claude-plans"

mkdir -p "$REPO_ROOT/book"
OUT="$REPO_ROOT/book/FoxML_Documentation.pdf"

# ── Ordered file list (reading order) ────────────────────────────
# Follows the nav structure from mkdocs.yml.
# Files that don't exist are silently skipped.
FILES=()
add() { [[ -f "$REPO_ROOT/$1" ]] && FILES+=("$REPO_ROOT/$1"); }

# Part 1: Home
add README.md

# Part 2: Executive Overview
add DOCS/00_executive/QUICKSTART.md
add DOCS/00_executive/GETTING_STARTED.md
add DOCS/00_executive/ARCHITECTURE_OVERVIEW.md
add DOCS/00_executive/DETERMINISTIC_TRAINING.md
add DOCS/00_executive/SYSTEM_REQUIREMENTS.md

# Part 3: Tutorials - Setup
add DOCS/01_tutorials/setup/README.md
add DOCS/01_tutorials/setup/INSTALLATION.md
add DOCS/01_tutorials/setup/ENVIRONMENT_SETUP.md
add DOCS/01_tutorials/setup/GPU_SETUP.md

# Part 3: Tutorials - Pipelines
add DOCS/01_tutorials/pipelines/README.md
add DOCS/01_tutorials/pipelines/FIRST_PIPELINE_RUN.md
add DOCS/01_tutorials/pipelines/CUSTOM_DATASETS.md
add DOCS/01_tutorials/pipelines/CUSTOM_FEATURES.md
add DOCS/01_tutorials/pipelines/DATA_LOADER_PLUGINS.md
add DOCS/01_tutorials/pipelines/DATA_PROCESSING_README.md
add DOCS/01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md
add DOCS/01_tutorials/pipelines/FEATURE_ENGINEERING_TUTORIAL.md

# Part 3: Tutorials - Training
add DOCS/01_tutorials/training/README.md
add DOCS/01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md
add DOCS/01_tutorials/training/MODEL_TRAINING_GUIDE.md
add DOCS/01_tutorials/training/AUTO_TARGET_RANKING.md
add DOCS/01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md
add DOCS/01_tutorials/training/WALKFORWARD_VALIDATION.md
add DOCS/01_tutorials/training/EXPERIMENTS_OPERATIONS.md

# Part 3: Tutorials - Configuration
add DOCS/01_tutorials/configuration/README.md
add DOCS/01_tutorials/configuration/CONFIG_BASICS.md
add DOCS/01_tutorials/configuration/ADVANCED_CONFIG.md
add DOCS/01_tutorials/configuration/CONFIG_EXAMPLES.md
add DOCS/01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md

# Part 4: Reference - API
add DOCS/02_reference/api/README.md
add DOCS/02_reference/api/CLI_REFERENCE.md
add DOCS/02_reference/api/CONFIG_SCHEMA.md
add DOCS/02_reference/api/DATA_PROCESSING_API.md
add DOCS/02_reference/api/INTELLIGENT_TRAINER_API.md
add DOCS/02_reference/api/MODULE_REFERENCE.md

# Part 4: Reference - Configuration
add DOCS/02_reference/configuration/README.md
add DOCS/02_reference/configuration/CONFIG_LOADER_API.md
add DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md
add DOCS/02_reference/configuration/ENVIRONMENT_VARIABLES.md
add DOCS/02_reference/configuration/MODEL_CONFIGURATION.md
add DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md
add DOCS/02_reference/configuration/RUN_IDENTITY.md
add DOCS/02_reference/configuration/TRAINING_PIPELINE_CONFIGS.md
add DOCS/02_reference/configuration/USAGE_EXAMPLES.md

# Part 4: Reference - Data
add DOCS/02_reference/data/README.md
add DOCS/02_reference/data/COLUMN_REFERENCE.md
add DOCS/02_reference/data/DATA_FORMAT_SPEC.md
add DOCS/02_reference/data/DATA_SANITY_RULES.md

# Part 4: Reference - Models
add DOCS/02_reference/models/README.md
add DOCS/02_reference/models/MODEL_CATALOG.md
add DOCS/02_reference/models/MODEL_CONFIG_REFERENCE.md
add DOCS/02_reference/models/TRAINING_PARAMETERS.md

# Part 4: Reference - Systems
add DOCS/02_reference/systems/README.md
add DOCS/02_reference/systems/PIPELINE_REFERENCE.md

# Part 4: Reference - Trading
add DOCS/02_reference/trading/README.md
add DOCS/02_reference/trading/ALPACA_CONFIGURATION.md
add DOCS/02_reference/trading/IBKR_CONFIGURATION.md
add DOCS/02_reference/trading/TRADING_MODULES.md

# Part 4: Reference - Training Routing
add DOCS/02_reference/training_routing/README.md
add DOCS/02_reference/training_routing/END_TO_END_FLOW.md
add DOCS/02_reference/training_routing/ONE_COMMAND_TRAINING.md
add DOCS/02_reference/training_routing/TWO_STAGE_TRAINING.md

# Part 5: Technical - Research
add DOCS/03_technical/research/README.md
add DOCS/03_technical/research/FEATURE_IMPORTANCE_METHODOLOGY.md
add DOCS/03_technical/research/INTELLIGENCE_LAYER.md
add DOCS/03_technical/research/TARGET_DISCOVERY.md
add DOCS/03_technical/research/VALIDATION_METHODOLOGY.md

# Part 5: Technical - Implementation
add DOCS/03_technical/implementation/README.md
add DOCS/03_technical/implementation/PARALLEL_EXECUTION.md
add DOCS/03_technical/implementation/PERFORMANCE_OPTIMIZATION.md
add DOCS/03_technical/implementation/REPRODUCIBILITY_API.md
add DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md
add DOCS/03_technical/implementation/TRAINING_OPTIMIZATION_GUIDE.md

# Part 5: Technical - Design
add DOCS/03_technical/design/README.md
add DOCS/03_technical/design/ARCHITECTURE_DEEP_DIVE.md
add DOCS/03_technical/design/contracts.md

# Part 5: Technical - Trading
add DOCS/03_technical/trading/README.md
add DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md
add DOCS/03_technical/trading/architecture/OPTIMIZATION_ARCHITECTURE.md
add DOCS/03_technical/trading/implementation/LIVE_TRADING_INTEGRATION.md
add DOCS/03_technical/trading/operations/OPERATIONS_GUIDE.md

# Part 5: Technical - Testing
add DOCS/03_technical/testing/README.md
add DOCS/03_technical/testing/TESTING_PLAN.md
add DOCS/03_technical/testing/TESTING_SUMMARY.md

# Part 5: Technical - Operations
add DOCS/03_technical/operations/README.md
add DOCS/03_technical/operations/SYSTEMD_DEPLOYMENT.md
add DOCS/03_technical/operations/JOURNALD_LOGGING.md

# Part 5: Technical - Benchmarks
add DOCS/03_technical/benchmarks/README.md
add DOCS/03_technical/benchmarks/DATASET_SIZING.md
add DOCS/03_technical/benchmarks/MODEL_COMPARISONS.md
add DOCS/03_technical/benchmarks/PERFORMANCE_METRICS.md

# Part 6: Module Documentation
add CONFIG/README.md
add TRAINING/training_strategies/README.md
add TRAINING/ranking/predictability/README.md
add LIVE_TRADING/README.md
add LIVE_TRADING/DOCS/architecture/SYSTEM_ARCHITECTURE.md
add LIVE_TRADING/DOCS/architecture/PIPELINE_STAGES.md
add LIVE_TRADING/DOCS/components/MODEL_INFERENCE.md
add LIVE_TRADING/DOCS/components/RISK_MANAGEMENT.md
add DASHBOARD/README.md
add DASHBOARD/OVERVIEW.md
add DASHBOARD/FEATURES.md
add DATA_PROCESSING/README.md

# Part 7: Developer Guide
add CLAUDE.md
add INTEGRATION_CONTRACTS.md
add CODE_REVIEW_SUMMARY.md

# Part 8: Internal Reference (key docs only)
add INTERNAL/docs/README.md
add INTERNAL/docs/INTERNAL_HANDBOOK.md
add INTERNAL/docs/references/SST_SOLUTIONS.md
add INTERNAL/docs/references/DETERMINISTIC_PATTERNS.md
add INTERNAL/docs/references/EXCEPTIONS_MATRIX.md
add INTERNAL/docs/foundations/MATHEMATICAL_FOUNDATIONS.md

# Part 9: Appendices
add CHANGELOG.md

echo "Building PDF from ${#FILES[@]} documents..."

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Error: No documents found!"
    exit 1
fi

pandoc "${FILES[@]}" \
    --toc \
    --toc-depth=3 \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V documentclass=report \
    -V title="FoxML Core Documentation" \
    -V author="FoxML Team" \
    -V date="$(date +%Y-%m-%d)" \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    --highlight-style=tango \
    --resource-path="$REPO_ROOT" \
    -o "$OUT"

echo ""
echo "PDF built at: $OUT"
echo "Pages: $(pdfinfo "$OUT" 2>/dev/null | grep Pages | awk '{print $2}' || echo 'unknown')"
