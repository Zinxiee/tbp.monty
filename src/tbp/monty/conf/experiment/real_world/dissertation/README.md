# Dissertation Experiments — Real-World Monty

Dissertation objective:

> "To expand upon and stress-test the Thousand Brains Project, designing and constructing a system capable of multimodal sensing and movement to enable **learning and recognition** of real-world objects in a **human-like way**."

---

## Hardware

| Agent | Hardware | Sensor | Working distance |
|---|---|---|---|
| Surface | Ufactory Lite 6 robot arm | Maixsense A010 ToF (100×100, 8-bit, 70°×60°) | 12 cm |
| Distant | ZED 2i stereo camera | RGBD (110°×70°, 2.12 mm focal) | ~50–100 cm |

---

## Object Set

Choose **5–7 physical objects** roughly mug/Rubik's-cube sized. Suggested categories:

| ID | Object | Category |
|---|---|---|
| O1 | Mug (TBP reference object) | Clearly distinct |
| O2 | Rubik's cube (or similar) | Clearly distinct |
| O3 | Cylindrical can | Moderately distinct |
| O4 | Small rectangular box | Moderately distinct |
| O5 | Second mug (different handle) | Confusable with O1 |
| O6 | Cube with rounded corners | Confusable with O2 (optional) |
| O7 | Object of your choice | (optional) |

### Orientation reference

Fix 5 orientations using a **centre mark on the table** and a **reference photo for each**:

| Label | Description |
|---|---|
| ORI0 | Canonical (training orientation) |
| ORI1 | 90° yaw rotation |
| ORI2 | 180° yaw rotation |
| ORI3 | 270° yaw rotation |
| ORI4 | ~30° off-axis tilt (forward or sideways) - Changed to unusual/different orientation |

Take a photograph of each (object × orientation) before the first run and commit to `benchmarks/dissertation/orientations/`.

### Repeat policy

Run **at least 3 episodes per object per orientation** before reporting any aggregate number. A single 100% or 0% episode is not evidence.

---

## Experiment Summary

| # | Config | Agent | Train/Eval | Supervised | New model? |
|---|---|---|---|---|---|
| 1 | `exp1_surface_train` | Surface | Train | Yes | Yes → save checkpoint |
| 1 | `exp1_surface_eval` | Surface | Eval | — | Load Exp 1 surface ckpt |
| 1 | `exp1_distant_train` | Distant | Train | Yes | Yes → save checkpoint |
| 1 | `exp1_distant_eval` | Distant | Eval | — | Load Exp 1 distant ckpt |
| 1 | `zed_supervised_eval` (stock) | Distant | Eval | — | TBP pretrained (mug only) |
| 2 | `exp2_surface_eval_rot` | Surface | Eval | — | Load Exp 1 surface ckpt |
| 2 | `exp2_distant_eval_rot` | Distant | Eval | — | Load Exp 1 distant ckpt |
| 3 | Analysis only | Both | — | — | Uses Exp 1+2 CSVs |
| 4 | `exp4_distant_continual` | Distant | Train | No | Fresh (unsupervised) |
| 4b | `exp4b_surface_continual` *(opt)* | Surface | Train | No | Fresh (unsupervised) |
| 5 | `exp5_surface_to_distant_eval` | Distant | Eval | — | Load Exp 1 surface ckpt |
| 5 | `exp5_distant_to_surface_eval` | Surface | Eval | — | Load Exp 1 distant ckpt |
| 6 | `exp6_surface_similar_eval` *(opt)* | Surface | Eval | — | Load Exp 1 surface ckpt |
| 6 | `exp6_distant_similar_eval` *(opt)* | Distant | Eval | — | Load Exp 1 distant ckpt |

---

## Experiment 1 — Baseline Feasibility

**Research question:** Can Monty learn and recognise real-world objects at all?

**Protocol:**

1. Place O1–O7 one at a time in ORI0.
2. **Surface Agent training:**
   ```bash
   python run.py experiment=real_world/dissertation/exp1_surface_train
   ```
   - Confirm object placement when prompted. One episode per object.
   - Note checkpoint path: `outputs/exp1_surface_train/0/model.pt`

3. **Surface Agent eval:**
   ```bash
   python run.py experiment=real_world/dissertation/exp1_surface_eval \
     experiment.config.model_name_or_path=outputs/exp1_surface_train/0/model.pt
   ```
   - Place each object in ORI0 when prompted. ≥3 episodes per object.

4. **Distant Agent training:**
   ```bash
   python run.py experiment=real_world/dissertation/exp1_distant_train
   ```
   - Scene picker: type `capture` to record a ZED RGBD frame, then select it.
   - Note checkpoint: `outputs/exp1_distant_train/0/model.pt`

5. **Distant Agent eval:**
   ```bash
   python run.py experiment=real_world/dissertation/exp1_distant_eval \
     experiment.config.model_name_or_path=outputs/exp1_distant_train/0/model.pt
   ```

6. **TBP pretrained sanity check (mug O1 only):**
   ```bash
   python run.py experiment=real_world/zed_supervised_eval
   ```
   This uses the TBP `surf_agent_1lm_numenta_lab_obj` model. Records whether the
   TBP mug model generalises to your physical mug as a sim-to-real baseline.

**Metrics:** `percent_correct`, `avg_rotation_error`, `avg_num_lm_steps` per agent.

**Analysis quick-check:**
```bash
python -c "
from tbp.monty.frameworks.utils.logging_utils import print_overall_stats
print_overall_stats('outputs/exp1_surface_eval/eval_stats.csv')
print_overall_stats('outputs/exp1_distant_eval/eval_stats.csv')
"
```

---

## Experiment 2 — Pose / Rotation Invariance

**Research question:** Does recognition accuracy hold when objects are placed at orientations not seen during training?

**Protocol:**

Run **exp2_surface_eval_rot** and **exp2_distant_eval_rot** once per non-canonical orientation (ORI1–ORI4), overriding `run_name`:

```bash
# ORI1 — 90° yaw
python run.py experiment=real_world/dissertation/exp2_surface_eval_rot \
  experiment.config.model_name_or_path=outputs/exp1_surface_train/0/model.pt \
  experiment.config.logging.run_name=exp2_surface_eval_rot1

python run.py experiment=real_world/dissertation/exp2_distant_eval_rot \
  experiment.config.model_name_or_path=outputs/exp1_distant_train/0/model.pt \
  experiment.config.logging.run_name=exp2_distant_eval_rot1

# Repeat for rot2, rot3, rot4 — change run_name each time
```

**Metrics:** Compare `percent_correct` and `avg_rotation_error` across ORI0–ORI4.

**Expected finding:** If the system is pose-invariant, accuracy should stay high and `avg_rotation_error` should remain low even at unseen orientations.

---

## Experiment 3 — Multimodal Comparison & Complementarity

**Research question:** Which modality (ToF vs. RGBD) works better for which type of object, and why?

**Protocol:** Analysis-only pass over Exp 1 and Exp 2 CSVs. No new hardware run.

1. Aggregate `percent_correct` and `avg_num_lm_steps` per object per agent from Exp 1.
2. Classify each object:
   - **ToF-favourable**: matte surface, thick geometry (good curvature signal).
   - **ZED-favourable**: rich colour/texture, smooth or reflective surface.
   - **Ambiguous**: mixed properties.
3. Record wall-clock episode duration (Surface Agent includes robot motion + settle).

**Output for report:** Side-by-side agent comparison table + per-object breakdown.

---

## Experiment 4 — Continual (Human-like) Unsupervised Learning

**Research question:** Can Monty learn new objects incrementally without labels, while retaining models of previously seen objects?

**Protocol:**

Use 5 objects: O1–O5. Presentation sequence (10 episodes, no labels):

```
Episode: 1    2    3    4    5    6    7    8    9    10
Object:  O1   O2   O3   O4   O5   O1   O3   O5   O2   O4
Type:    learn learn learn learn learn recall recall recall recall recall
```

```bash
python run.py experiment=real_world/dissertation/exp4_distant_continual
```

Scene picker: select/capture one ZED scene per episode in the sequence above.

**Analysis:**
```bash
python -c "
from tbp.monty.frameworks.utils.logging_utils import print_unsupervised_stats
print_unsupervised_stats('outputs/exp4_distant_continual/train_stats.csv')
"
```

**Key metrics:**
- `mean_objects_per_graph` — ideally → 1 (each graph = one object)
- `mean_graphs_per_object` — ideally → 1 (each object = one graph)
- Episode-by-episode hit/miss for return visits (episodes 6–10)

---

## Experiment 4b (optional) — Surface Agent Continual Learning

**Research question:** Does human-like continual learning hold under active sensorimotor exploration?

**Protocol:** 3 objects, 5 episodes, Surface Agent.

```
Episode: 1    2    3    4    5
Object:  O1   O2   O3   O1   O3
Type:    learn learn learn recall recall
```

```bash
python run.py experiment=real_world/dissertation/exp4b_surface_continual
```

**Hardware budget:** ~1 hour. Run only if session time permits.

**Comparison:** Match Exp 4 metrics for the same 3 objects to determine whether active exploration improves or matches passive-capture continual learning.

---

## Experiment 5 — Cross-Agent Model Transfer

**Research question:** Are Monty's object models sensor-agnostic — can a model built with one sensor be recognised through the other?

**Protocol:**

**Direction 1: Surface-trained model → Distant Agent inference**
```bash
python run.py experiment=real_world/dissertation/exp5_surface_to_distant_eval \
  experiment.config.model_name_or_path=outputs/exp1_surface_train/0/model.pt
```

**Direction 2: Distant-trained model → Surface Agent inference**
```bash
python run.py experiment=real_world/dissertation/exp5_distant_to_surface_eval \
  experiment.config.model_name_or_path=outputs/exp1_distant_train/0/model.pt
```

Build the 2×2 transfer matrix for the report:

| | Eval: Surface Agent | Eval: Distant Agent |
|---|---|---|
| **Train: Surface Agent** | Exp 1 (baseline) | Exp 5 (S→D) |
| **Train: Distant Agent** | Exp 5 (D→S) | Exp 1 (baseline) |

**Note:** Zero accuracy on Exp 5 is a valid negative finding. Document the
architectural reason (e.g. missing HSV features in surface model) and frame as
a known limitation with engineering implications for future work.

---

## Experiment 6 (optional) — Similar Object Discrimination

**Research question:** Can the system distinguish geometrically similar objects?

**Protocol:** Run Exp 1 eval configs with only the confusable object subset (e.g. O1 + O5, O2 + O6). Use Exp 1 checkpoints — no retraining.

```bash
# Surface Agent
python run.py experiment=real_world/dissertation/exp6_surface_similar_eval \
  experiment.config.model_name_or_path=outputs/exp1_surface_train/0/model.pt

# Distant Agent
python run.py experiment=real_world/dissertation/exp6_distant_similar_eval \
  experiment.config.model_name_or_path=outputs/exp1_distant_train/0/model.pt
```

**Metrics:** `percent_correct` and `percent_confused` on the confusable pair. Which modality disambiguates better?

---

## Config inheritance

All dissertation configs inherit from the existing real-world configs:

```
exp1/2/5/6_surface_*  →  /experiment/real_world/lite6_maixsense_unsupervised
exp1/2/5/6_distant_*  →  /experiment/real_world/zed_supervised_eval (eval)
                      →  /experiment/real_world/zed_continual_manual_train (train)
exp4b_surface_*       →  lite6_maixsense_unsupervised + noresetevidence_1lm override
exp4_distant_*        →  zed_continual_manual_train (already uses noresetevidence_1lm)
```

These parent configs encode tuned hardware defaults (12 cm working distance, depth
burst averaging, goal dispatch, semantic bounds filtering). Do not duplicate them.

To inspect the fully-resolved config for any experiment:
```bash
python run.py experiment=real_world/dissertation/exp1_surface_train --cfg job
```
Commit the output to `benchmarks/dissertation/resolved_configs/` for reproducibility.

---

## Output structure

Each run writes to `outputs/<run_name>/`:
```
outputs/exp1_surface_train/
  0/
    model.pt            ← checkpoint for downstream experiments
    exp_state_dict.pt
    config.pt
  train_stats.csv       ← one row per episode
  eval_stats.csv        ← one row per episode (if do_eval: true)
```

Aggregate per-experiment CSVs into `benchmarks/dissertation/` for report analysis.

---

## Recommended run order

```
1. exp1_surface_train        (prerequisite for all surface eval experiments)
2. exp1_distant_train        (prerequisite for all distant eval experiments)
3. exp1_surface_eval         (Exp 1 baseline)
4. exp1_distant_eval         (Exp 1 baseline)
   zed_supervised_eval       (TBP sanity check, mug only)
5. exp2_surface_eval_rot×4   (Exp 2 — one run per orientation)
6. exp2_distant_eval_rot×4   (Exp 2 — one run per orientation)
7. exp4_distant_continual    (Exp 4 — independent, no prior checkpoint needed)
8. exp5_surface_to_distant_eval
9. exp5_distant_to_surface_eval
[optional]
10. exp4b_surface_continual
11. exp6_surface_similar_eval
12. exp6_distant_similar_eval
```
