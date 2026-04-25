# Attribution & AI Tool Usage

This document details the use of AI development tools in the Chapel Covers project.

## AI Tool Usage Summary

**Primary Tool:** Claude (Anthropic)  
**Usage Context:** Scaffolding, debugging, and optimization of ML pipeline code  
**Percentage of Code:** ~40% Claude-generated (scaffolded), 60% hand-written/modified by developer  

---

## Detailed Breakdown by Component

### 1. LoRA Training Script (`scripts/lora_train.py`)

**AI Generated:** 80% (initial scaffold)  
**Developer Modified:** 20% (debugging, fixing errors, testing)

**What Claude Generated:**
- Overall training loop structure
- DDPM loss computation with noise scheduling
- VAE encoding pipeline
- Text tokenization from prompts
- Optimizer and learning rate setup
- Model checkpointing and saving logic

**What Developer Had to Fix:**
- Initial version had incorrect UNet input signature (missing timestep & encoder_hidden_states)
- Had to rewrite training loop from simplified pixel reconstruction to proper diffusion training
- Fixed tensor operations for Mac MPS GPU compatibility
- Debugged and fixed PEFT/LoRA configuration issues
- Iterated through multiple test runs to get working version
- Added proper error handling and validation

**Key Lesson:** AI scaffolding was valuable for structure, but the actual ML implementation required deep understanding of diffusion training mechanics. Developer had to substantially rework the core training logic.

---

### 2. Flask Server with LoRA Support (`app/flask_server_lora.py`)

**AI Generated:** 70% (initial structure)  
**Developer Modified:** 30% (integration, testing, optimization)

**What Claude Generated:**
- Flask app initialization and routing structure
- Argument parsing for LoRA path selection
- Error handlers and response formatting
- File upload handling with security checks
- Base REST endpoint structure

**What Developer Had to Modify:**
- Integrated actual LoRA loading logic (Claude's initial approach wouldn't load safetensors format)
- Added genre refinement pipeline integration
- Connected prompt builder to generation pipeline
- Tested and debugged all API endpoints
- Added proper device detection for Mac GPU
- Optimized response serialization

**Notes:** Flask scaffolding from Claude was nearly production-ready. Most modifications were about integrating with actual ML components.

---

### 3. LoRA Integration Layer (`src/lora_integration.py`)

**AI Generated:** 60% (initial design)  
**Developer Modified:** 40% (implementation, testing, refinement)

**What Claude Generated:**
- Class structure extending CoverArtPipeline
- Method signatures for loading/unloading LoRA
- Documentation and docstrings
- Placeholder for LoRA blending logic

**What Developer Had to Implement:**
- Actual PEFT model loading from saved weights
- Fixed safetensors format handling
- Implemented proper error checking
- Added safety validation for model state
- Tested with actual trained LoRA weights
- Debugged PEFT integration with Hugging Face diffusers library

---

### 4. Prompt Builder Refinements (`src/prompt_builder.py`)

**AI Generated:** 40% (mapping dictionaries and structure)  
**Developer Modified:** 60% (tuning prompts, adding keywords, manual refinement)

**What Claude Generated:**
- Core mapping structure for refinement feedback
- Initial prompt templates for genres
- Basic validation logic for refinement inputs

**What Developer Had to Tune:**
- Manually adjusted all genre style descriptors through iterative testing
- Tested and refined refinement mappings (e.g., "darker", "black and white")
- Added missing genre descriptions and mood modifiers
- Iterated on prompt length and keyword placement
- Validated that prompts actually produce better images
- Added Duke aesthetic elements through manual prompt engineering

**Key Insight:** Prompt engineering is fundamentally creative and requires human judgment. Claude provided good structure, but developer had to manually tune every single prompt descriptor based on actual output quality.

---

### 5. Genre Refinement Heuristics (`src/pipeline.py`)

**AI Generated:** 50% (initial detection logic)  
**Developer Modified:** 50% (threshold tuning, validation, testing)

**What Claude Generated:**
- Method structure for `_refine_genre_with_features()`
- Basic if-statement logic for common misclassifications
- Documentation of detection strategy

**What Developer Had to Tune:**
- Manually tested on actual audio samples
- Adjusted tempo/energy/brightness thresholds based on test results
- Added validation rules and edge case handling
- Iterated multiple times to get good accuracy improvement
- Tested on misclassified examples (Classical→Pop, Rock→Blues)
- Balanced false positive rate vs. correction rate

---

## Code Quality & Understanding

### Developer's Contribution Beyond Code

The developer demonstrated understanding of the ML pipeline by:

1. **Debugging LoRA Training:** Identified that initial training loop was calling UNet incorrectly. Understood diffusion mechanics deeply enough to rewrite the training loop.

2. **Feature-Based Genre Refinement:** Designed and implemented logic to use extracted audio features to correct CNN misclassifications. Manually tested thresholds on actual audio samples.

3. **Prompt Engineering:** Manually tuned 50+ prompt variations. Tested generated images iteratively and refined descriptors based on visual output quality.

4. **LoRA Integration:** Debugged safetensors loading issues. Tested different LoRA scales and documented optimal values per genre.

5. **Error Analysis:** Identified CNN misclassifications and implemented solutions with validation.

---

## AI Tools NOT Used

- No Copilot or GitHub code completion during development
- No automated hyperparameter optimization
- No pre-built prompt optimization libraries
- No high-level "magic" libraries (all core ML hand-written or custom-integrated)

---

## File-by-File Attribution

| File | Claude % | Developer % |
|------|----------|-------------|
| `src/pipeline.py` | 40 | 60 |
| `src/prompt_builder.py` | 40 | 60 |
| `src/lora_integration.py` | 60 | 40 |
| `scripts/lora_train.py` | 80 | 20 |
| `app/flask_server_lora.py` | 70 | 30 |
| `src/model.py` | 0 | 100 |
| `src/preprocessing.py` | 0 | 100 |

---

**Summary:** ~45% of code from Claude scaffolding, ~55% hand-written or substantially modified. Developer retained full understanding of all components and made critical design/tuning decisions.
