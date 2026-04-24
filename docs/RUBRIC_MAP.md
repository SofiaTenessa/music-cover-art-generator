# Rubric Mapping

This document maps each rubric item you plan to claim to its evidence in the repository. Use this to write your `self_assessment_template.docx` at submission time.

## Machine Learning (73 pts cap, ≤15 items)

| # | Item (points) | Evidence location |
|---|---------------|-------------------|
| 1 | Processed audio with spectrograms (7) | `src/preprocessing.py` — `audio_to_log_mel`, mood features |
| 2 | Cross-modal generation (audio → image) (7) | `src/pipeline.py` — full audio-to-image pipeline |
| 3 | Multi-stage ML pipeline (7) | `src/pipeline.py` — CNN → prompt → Stable Diffusion |
| 4 | Ablation study 2+ design choices (7) | `docs/ablation_results.md` — e.g., augment vs none, dropout 0.2 vs 0.4 |
| 5 | Error analysis with failure cases (7) | `src/evaluate.py` — `find_hardest_examples`, confusion matrix |
| 6 | Compared multiple architectures (7) | `docs/comparison.md` — random baseline vs MFCC+LR vs CNN |
| 7 | Defined custom CNN (5) | `src/model.py` — `GenreCNN` |
| 8 | Used pretrained diffusion model (5) | `src/pipeline.py` — Stable Diffusion integration |
| 9 | Conditioning controls in diffusion (5) | `src/prompt_builder.py` — structured prompts + negative prompt |
| 10 | Data augmentation (5) | `src/dataset.py` — `SpecAugment` + ablation showing impact |
| 11 | Regularization ≥2 techniques (5) | `src/train.py` — dropout, L2 weight decay, early stopping |
| 12 | Hyperparameter tuning 3+ configs (5) | `notebooks/train_cnn_colab.ipynb` sweep cell + `docs/hyperparameter_search.md` |
| 13 | Web app deployment (10) | `app/streamlit_app.py` |
| 14 | Documented 2+ improvement iterations (5) | `docs/iterations.md` (fill this in as you iterate) |
| 15 | Solo project (10) | README "Individual Contributions" |

**Subtotal before cap: 102 pts. Capped at 73.**

## Following Directions (~15 pts)

- [x] Self-assessment with ≤15 items + evidence (3)
- [x] SETUP.md (1)
- [x] ATTRIBUTION.md (1)
- [x] requirements.txt (1)
- [x] README What it Does (1)
- [x] README Quick Start (1)
- [x] README Video Links (1) — fill in URLs before submission
- [x] README Evaluation (1) — fill in results after training
- [x] README Individual Contributions (1) — solo, noted in README
- [ ] Demo video correct length (2)
- [ ] Technical walkthrough correct length (2)

## Project Cohesion (15 pts)

- [x] README articulates unified project goal (3)
- [ ] Demo video explains why project matters (3)
- [x] Addresses meaningful research question (3) — cross-modal generation
- [ ] Walkthrough shows components work together (2)
- [ ] Clear progression: problem → approach → solution → evaluation (2)
- [ ] Design choices justified (2)
- [x] Evaluation metrics measure project objectives (2)
- [x] No superfluous components (2)
- [x] Clean codebase (2)

## Target total: 73 + 15 + 15 = 103 (max possible)

---

## To-do before submission

1. Run training, record metrics into `README.md` Evaluation section
2. Run ablation study — see `docs/ablation_results.md` (create this)
3. Run hyperparameter sweep — record in `docs/hyperparameter_search.md`
4. Generate ~20 example cover arts, save best in `outputs/examples/`
5. Record demo video (non-technical)
6. Record technical walkthrough
7. Fill in video links + Evaluation table in `README.md`
8. Write ATTRIBUTION.md "What I modified" / "What I had to debug" sections
9. Finalize self-assessment doc using this mapping
