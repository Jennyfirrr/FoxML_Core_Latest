# FoxML Core — ML Cross-Sectional Infrastructure

---

## License & Enforcement

**© 2025-2026 Fox ML Infrastructure LLC. All rights reserved.**
**U.S. Copyright Registration Case No. 1-15101732111** — Registered with the U.S. Copyright Office as a Literary Work.

**This software is dual-licensed:** [AGPL-3.0-or-later](LICENSE) **or** [Commercial](LICENSE-COMMERCIAL). If you use this software without complying with the AGPL (including the requirement to publish your source code for any network-accessible deployment) and without a commercial license, you are infringing copyright.

### Unauthorized Use — Settlement Terms

If you or your organization are found using FoxML without a valid license:

- **$250,000 USD flat fee** plus **1-10% of gross revenue** derived from use of the software, from date of first unauthorized use through settlement
- The Copyright Holder reserves the right to pursue **full statutory damages** under 17 U.S.C. Section 504 (up to **$150,000 per work** for willful infringement), **injunctive relief**, and **attorney's fees**

---

**FoxML is research-grade ML infrastructure with deterministic strict mode + full fingerprint lineage. It assumes prior experience with Python, Linux, and quantitative workflows. As always, some parts are experimental and subject to breakage as work continues.**

> ⚠️ **Disclaimer:** This software is provided for research and educational purposes only. It does not constitute financial advice, and no guarantees of returns or performance are made. Use at your own risk.

> 💻 **Interface:** This is a command-line and config-driven system. There is no graphical UI, web dashboard, or visual interface. All interaction is via YAML configuration files and Python scripts.

> **⚠️ WARNING: This software is buggy. Use at your own risk.**
>
> **Maintenance Status:** The maintenance status of this project is a grey area between maintained and unmaintained. It is highly subject to how I'm feeling at any given time. Updates and bug fixes may or may not happen depending on my current capacity and interest.
>
> **Known Issues:**
> - **IBKR module**: Lost during development. The IBKR API is difficult to work with and this module will likely not be continued.
> - **Alpaca module**: The Alpaca trading module is being redesigned. Legacy implementations are in `ARCHIVE/` for reference. The current execution engine is in `LIVE_TRADING/`.
>
> **License:** Dual-licensed — AGPL-3.0-or-later ([LICENSE](LICENSE)) or Commercial ([LICENSE-COMMERCIAL](LICENSE-COMMERCIAL))

> **🔍 Reproducibility & Auditability:** This system supports **bitwise deterministic runs** via strict mode (`bin/run_deterministic.sh`) for financial audit compliance. Bitwise determinism requires CPU-only execution, pinned dependencies, fixed thread env vars, and deterministic data ordering. Note: Not guaranteed across different CPUs/BLAS versions/kernels/drivers/filesystem ordering. See [Deterministic Runs](DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md).

> **🎯 Version 1.0 Definition:** See [FOXML_1.0_MANIFEST.md](DOCS/00_executive/product/FOXML_1.0_MANIFEST.md) for the capability boundary that defines what constitutes FoxML 1.0.

> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

---

FoxML Core is an ML infrastructure stack for cross-sectional and panel data, supporting both pooled cross-sectional training and symbol-specific (per-symbol) training modes. Designed for any machine learning applications requiring temporal validation and reproducibility. It provides a config-driven ML pipeline architecture designed for ML infra teams, data scientists, and researchers.

**Why This Exists:** Most ML repos are notebooks + scripts; FoxML is pipeline + audit artifacts. Designed to make research reproducible, comparable, and reviewable. This is infrastructure-first ML tooling, not a collection of example notebooks.

> **📊 Testing & Development:** All testing, validation, and development work is performed using **5-minute interval data**. The software supports various data intervals, but all tests, benchmarks, and development workflows use 5-minute bars as the standard reference.
>
> **Note on Data Intervals:** While the system is workable for all data sets, using different intervals requires edited configs. There is a partial implementation to handle different data intervals automatically, but it is not completely implemented or configured yet.
>
> **Note on Custom Data Sets:** For updating the feature set for custom data sets, you will need to manually edit the way that leaky features are determined (including regex patterns) to support new features if they do not follow the naming scheme that the DATA_PROCESSING files show. Automatic management of leaky feature detection for custom naming schemes was considered a later feature and not foundational, so it was never fully implemented.
>
> **Aspirational End Goal:** An eventual goal is to enable automatic discovery where you simply set the data path in a config file and the system would automatically discover features and targets, patch the registry, and configure everything without manual intervention. This is currently aspirational and not yet implemented.

Developed and maintained by **Jennifer Lewis**  
🔗 [LinkedIn](https://www.linkedin.com/in/jennifer-l-3434b3315)

**Research & Development Support:** If you would like to fund or assist research and development, please message me on [LinkedIn](https://www.linkedin.com/in/jennifer-l-3434b3315) or by email at jenn.lewis5789@gmail.com

---

## What You Get

- **3-stage intelligent pipeline**: Automated target ranking → feature selection → model training with config-driven routing decisions
- **Dual-view evaluation**: Supports both cross-sectional (pooled across symbols) and symbol-specific (per-symbol) training modes for comprehensive target analysis
- **Automatic routing**: Intelligently routes targets to cross-sectional, symbol-specific, or both training modes based on performance metrics and data characteristics
- **Task-aware evaluation**: Unified handling of regression (IC-based) and classification (AUC-based) targets with normalized skill scores
- **GPU acceleration** for target ranking, feature selection, and model training (LightGBM, XGBoost, CatBoost)
- **Bitwise deterministic runs** via strict mode (CPU-only, pinned dependencies, fixed thread env vars, deterministic data ordering) for financial audit compliance. Note: Not guaranteed across different CPUs/BLAS versions/kernels/drivers/filesystem ordering.
  - *Yes, outputs remain deterministic even while you're grinding OSRS, watching YouTube, or questioning your life choices at 3 AM. We tested it.*
- **Config-based usage** with minimal command-line arguments
- **Leakage detection system** with pre-training leak detection and auto-fix
- **Single Source of Truth (SST)** config system - all 20 model families use config-driven hyperparameters
- **Multi-model training systems** with 20 model families (LightGBM, XGBoost, CatBoost, MLP, CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer, RewardBased, QuantileLightGBM, NGBoost, GMMRegime, ChangePoint, FTRLProximal, VAE, GAN, Ensemble, MetaLearning, MultiTask) - GPU-accelerated where supported
- **Local metrics tracking** - By default, runs are local-only; no data is sent externally. All metrics stored locally for reproducibility.

---

## Quick Start (30 Seconds)

```bash
# pip install (recommended — works in any virtualenv)
git clone <repository-url>
cd FoxML
pip install -e .             # Core (tree models, pipeline, config)
pip install -e ".[gpu]"      # + PyTorch/TensorFlow for neural families
pip install -e ".[all]"      # Everything

# Run a quick test
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "test_install"

# Check results
ls RESULTS/runs/*/globals/config.resolved.json
```

**Conda install** (for full GPU/CUDA source builds):
```bash
bash bin/install.sh
conda activate trader
```

See [Quick Start Guide](DOCS/00_executive/QUICKSTART.md) for full setup.

---

## Key Concepts

**Cross-sectional vs Symbol-specific:**
- **Cross-sectional**: Pooled training across all symbols (learns patterns common across the universe)
- **Symbol-specific**: Per-symbol training (learns patterns unique to each symbol)

**Pipeline Stages:**
- **Target Ranking**: Ranks targets by predictability using multiple model families
- **Feature Selection**: Selects optimal features per target using importance analysis
- **Training**: Trains final models with routing decisions (cross-sectional, symbol-specific, or both)

**Determinism Modes:**
- **Strict mode**: Bitwise deterministic (CPU-only, single-threaded, pinned deps) - use `bin/run_deterministic.sh`
- **Best-effort mode**: Seeded but may vary (allows GPU, multi-threading) - default behavior

**Fingerprints vs RunIdentity:**
- **Fingerprints**: SHA256 hashes of individual components (data, config, features, targets)
- **RunIdentity**: Complete run signature combining all fingerprints for full traceability

---

## Fingerprinting & Reproducibility

- **3-stage pipeline architecture**: TARGET_RANKING (ranks targets by predictability) → FEATURE_SELECTION (selects optimal features) → TRAINING (trains models with routing decisions)
- **Dual-view support**: Each stage evaluates targets in both CROSS_SECTIONAL (pooled) and SYMBOL_SPECIFIC (per-symbol) views for comprehensive analysis
- **SHA256 fingerprinting** for all pipeline components — data, config, features, targets, splits, hyperparameters, and routing decisions
- **RunIdentity system** with two-phase construction (partial → finalized) and strict/replicate key separation for audit-grade traceability
- **Diff telemetry** — automatic comparison with previous runs, distinguishing true regressions from acceptable nondeterminism
- **Feature importance snapshots** with cross-run stability analysis and drift detection
- **Stage-scoped output layout** — target-first directory organization with `stage=TARGET_RANKING/`, `stage=FEATURE_SELECTION/`, `stage=TRAINING/` separation for human-readable auditability
- **Cohort-based reproducibility** — each (target, view, universe, cohort) combination gets its own snapshot with full fingerprint lineage

**For detailed capabilities:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

---

## System Requirements

FoxML Core is designed to scale from laptop experiments to production-scale deployments. Hardware requirements scale with your data size and pipeline configuration.

**Small Experiments (Laptop-Friendly):**
- **RAM**: 16-32GB minimum
- **Use Cases**: Small sample sizes, limited universe sizes, reduced feature counts
- Suitable for development, testing, and small-scale research

**Production / Ideal Configuration:**
- **RAM**: 128GB minimum, 512GB-1TB recommended for best performance
- **Use Cases**: Large sample counts, extensive universe sizes, full feature sets
- Enables full pipeline execution without memory constraints

**Scaling Factors:**
- **Sample count**: More samples require more memory for data loading and model training
- **Universe size**: Larger symbol universes increase memory usage proportionally
- **Feature count**: Feature count directly affects hardware usage (more features = more memory and compute)

**Universe Batching:**
The pipeline supports batching large universes across multiple runs. While batching works, **running with as few batches as possible is ideal for best results** - it enables better cross-sectional analysis and more comprehensive feature selection across the full universe.

**CPU Recommendations:**
- **Stable clocks**: Disable turbo boost/overclocking features for stability and consistency
- **Undervolting**: Slight undervolting is recommended for stability (reduces thermal throttling and power fluctuations)
- **Newer CPUs**: Generally perform better due to improved instruction sets and efficiency
- **Core count**: More cores are beneficial, but some operations are single-threaded, so core count only helps with parallel aspects of the pipeline
- **Base clock speed**: Faster base clocks improve performance across all operations
- **Best practice**: Disable turbo boost features and use stable, consistent clock speeds for reproducible results

**GPU Considerations:**
- **VRAM dependent**: GPU performance is primarily limited by available VRAM rather than compute cores
- **Non-determinism**: GPU operations introduce slight non-determinism (generally within acceptable tolerances) due to parallel floating-point arithmetic where operation ordering is not guaranteed
- **Strict mode**: For bitwise deterministic runs, GPU is automatically disabled for tree models (LightGBM, XGBoost, CatBoost) in strict mode
- **Best practice**: More VRAM and newer GPU architectures generally provide better performance when GPU acceleration is enabled

---

## Domain Focus

FoxML Core is **general-purpose ML cross-sectional infrastructure** for panel data and time-series workflows. The architecture provides domain-agnostic primitives with built-in safeguards (leakage detection, temporal validation, feature registry systems).

**Domain Applications:** Financial time series, IoT sensor data, healthcare, clickstream analytics, and any panel data with temporal structure.

**For detailed domain information:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

---

## Intended Use

### Appropriate Use Cases
- Research and experimentation
- ML workflow and architecture study
- Open-source projects
- Internal engineering reference
- Production deployments (when pinned to tagged releases / frozen configs)

### Not Appropriate For
- Unmodified production deployment without proper testing and validation
- Production use of `main` branch (not stable for production)

**Production Note:** Production is supported when pinned to tagged releases / frozen configs; `main` branch is not stable for production use.

**FoxML Core provides ML infrastructure and architecture, not domain-specific applications or pre-built solutions.**

---

## Privacy & Security

**Privacy:** This codebase contains no networking or phone-home functions or processes. It can be operated completely privately and offline. All operations are local-only by default.

**Security:** This software does not include built-in security features. However, if you understand how to use it properly, you can avoid most major security issues such as:
- Sabotaged data (through proper data validation and source verification)
- Library vulnerabilities (through dependency management and security scanning)
- Other common security pitfalls (through proper configuration and operational practices)

The run identity and hash IDs can serve as a security audit mechanism to detect if anything changes from the same data inputs, but this security auditing capability was not a development priority. **You should only feed this system trusted data** - do not rely on the fingerprinting system as a primary security control.

Users are responsible for implementing appropriate security measures for their deployment environment.

---

## Getting Started

**New users start here:**
- **[Quick Start](DOCS/00_executive/QUICKSTART.md)** - Get running in 5 minutes
- **[Getting Started](DOCS/00_executive/GETTING_STARTED.md)** - Complete onboarding guide
- **[Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)** - System at a glance
- **[Deterministic Runs](DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md)** - Bitwise reproducible runs for audit compliance

**Complete documentation:**
- **[Documentation Index](DOCS/INDEX.md)** - Full documentation navigation
- **[Tutorials](DOCS/01_tutorials/)** - Step-by-step guides
- **[Reference Docs](DOCS/02_reference/)** - Technical reference
- **[Technical Appendices](DOCS/03_technical/)** - Deep technical topics

---

## Understanding Design Choices

**Documentation Philosophy:** This codebase contains more documentation and comments than actual code. Design choices, architectural decisions, and structural rationale are extensively documented in:
- Inline code comments explaining why, not just what
- `DOCS/` directory with comprehensive technical documentation
- `INTERNAL/` directory with development references and design audits
- Configuration files with detailed comments explaining options

**If you have questions about design choices:**
- **Read the documentation** - Most design decisions are documented in `DOCS/` or explained in code comments
- **Use RAG or LLM tools** - The extensive documentation is well-suited for retrieval-augmented generation. Use an LLM with RAG capabilities to summarize design choices, architectural patterns, and decision rationale from the documentation corpus
  - **Important:** When asking for summarizations, prioritize most recently edited files. Each file contains edit metadata (modification timestamps) that can be used to sort by recency and focus on current design decisions rather than legacy choices
- **Search the codebase** - Many design choices are explained in comments near the relevant code

**Note on Documentation Currency:** Some documentation may be dated or reflect legacy design choices that have since evolved. The code itself (and its comments) is the most current source of truth. When using RAG/LLM tools, prioritize recently edited files to ensure you're getting current design rationale rather than outdated documentation.

**Working with AI Coding Agents:** Experience has shown that AI coding performance and problem-solving capability degrades when too many constraints are placed in rule files. Basic outlines and high-level guidance generally provide better results than exhaustive rule sets. For some tasks, explicitly telling coding agents to ignore certain rules or constraints can be a more effective approach than trying to encode every edge case. The rule files in this repository are intended as guidelines and reference points, not rigid constraints that must be followed in every context.

**Internal Planning Documents:** This repository includes internal planning documents, design work-in-progress, and development references (such as those in `INTERNAL/` and `.claude/plans/`). This is an intentional decision to provide transparency into the design process, architectural thinking, and problem-solving approach. The `.claude/plans/` files in particular provide insight into how the planning process worked when implementing new features or refactors - showing the thought process, trade-offs considered, and approach taken. These documents may include incomplete thoughts, abandoned approaches, or evolving designs - they are included as-is to show the reasoning behind decisions, not as polished final documentation.

The codebase prioritizes explicit communication of design intent over brevity. When in doubt, the documentation and comments are the source of truth for understanding why things are structured the way they are.

---

## Repository Structure

```
FoxML_Core/
├── DATA_PROCESSING/       (Pipelines & feature engineering)
├── TRAINING/              (Model training & research workflows)
├── CONFIG/                (Configuration management system)
├── DOCS/                  (Technical documentation)
└── SCRIPTS/               (Utilities & tools)
```

**For detailed structure:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

---

## Reporting Issues

For bug reports, feature requests, or technical issues:

- **GitHub Issues**: [Open an issue](https://github.com/Fox-ML-infrastructure/FoxML_Core/issues) (preferred for bug reports and feature requests)
- **Email**: jenn.lewis5789@gmail.com (for security issues, sensitive bugs, or private inquiries)

For questions or organizational engagements:  
**jenn.lewis5789@gmail.com**

---
