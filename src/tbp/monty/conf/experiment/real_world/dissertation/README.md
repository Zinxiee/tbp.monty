# Dissertation Experiments — Real-World Monty

Dissertation objective:

> "To expand upon and stress-test the Thousand Brains Project, designing and constructing a system capable of multimodal sensing and movement to enable **learning and recognition** of real-world objects in a **human-like way**."

This README is the executable guide for the dissertation experiment battery. Every
experiment is a numbered protocol with the same template (Research question →
Prereqs → Objects → Orientations → Protocol → Outputs → Metrics → Pass criteria) so
there is no ambiguity about what to run, what gets produced, and how to read it.

The surface agent results were collected before the dissertation YAML protocol was
finalised; they are presented as Experiment 1a (training-feasibility, single-episode
unsupervised) plus a sensor-noise / repeatability **Limitations chapter**. All
recognition-accuracy claims come from the distant agent.

---

## Hardware

| Agent | Hardware | Sensor | Working distance |
|---|---|---|---|
| Surface | Ufactory Lite 6 robot arm | Maixsense A010 ToF (100×100, 8-bit, 70°×60°) | 12 cm |
| Distant | ZED 2i stereo camera | RGBD (110°×70°, 2.12 mm focal) | ~50–100 cm |

---

## Object Set

| ID | Object tag | Distant | Surface | Notes |
|---|---|---|---|---|
| O1 | `tbp_mug` | ✓ | — | TBP reference object; similar-pair partner |
| O2 | `sw_mug` | ✓ | — | Similar-pair partner |
| O3 | `tea_tin` | ✓ | — | |
| O4 | `mc_fox` | ✓ | ✓ (×1) | Matte plastic fox |
| O5 | `hexagons` | ✓ | — | |
| O6 | `cap` | ✓ | ✓ (×1) | Black cap |
| O7 | `washbag` | ✓ | ✓ (×3 repeats) | |
| O8 | `cardboard_box` | — | ✓ (×3 repeats) | Surface-only sensor-noise probe |

**Similar pair (Exp 6):** `tbp_mug` ↔ `tea_tin` — geometrically similar mugs, distinguished by surface texture / colour.

### Orientation reference (distant agent)

Five orientations are pre-captured on a fixed table mark; reference photos live in
`benchmarks/dissertation/orientations/`:

| Label | Description |
|---|---|
| ORI0 | Canonical (training orientation) |
| ORI1 | 90° yaw rotation |
| ORI2 | 180° yaw rotation |
| ORI3 | 270° yaw rotation |
| ORI4 | ~30° off-axis tilt |

Orientation only applies to distant-agent experiments. The surface agent runs are at a
single canonical pose per object.

### Repeat policy

- **Distant:** ≥3 episodes per object per orientation before reporting any aggregate.
- **Surface:** training-only, 1 episode per object except `washbag` and `cardboard_box`
  which were repeated 3× each as a sensor-noise probe. No further surface runs will
  be performed.

---

## Experiment Summary

| # | Step | Agent | Action | Status | Output dir |
|---|---|---|---|---|---|
| 1a | Surface unsupervised training (×8) | Surface | Train | **Done** | `experiment_results/real_world_lite6_maixsense_unsupervised_*/` |
| 1b | `exp1_distant_train` | Distant | Supervised train | **Done** | `experiment_results/exp1_distant_train/pretrained/` |
| 1c | `exp1_distant_eval` | Distant | Eval | **Done** | `experiment_results/exp1_distant_eval/` |
| 1d | `zed_supervised_eval` (TBP mug sanity) | Distant | Eval | Optional | `experiment_results/zed_supervised_eval/` |
| 2 | `exp2_distant_eval_rot` × {ORI1..ORI4} | Distant | Eval | Pending | `…_rot1..4/` |
| 3 | Modality discussion (analysis-only) | — | — | After Exp 1+2 | `benchmarks/dissertation/analysis/exp3/` |
| 4 | `exp4_distant_continual` | Distant | Train (no labels) | Pending | `experiment_results/exp4_distant_continual/` |
| 5 | Cross-agent transfer **discussion-only** | — | — | Not executed | `benchmarks/dissertation/analysis/exp5/` |
| 6 | `exp6_distant_similar_eval` | Distant | Eval | **Done** | `experiment_results/exp6_distant_similar_eval/` |
| L | Surface sensor-noise / repeatability study | Surface | Analysis-only | **Done** | `benchmarks/dissertation/analysis/surface_unsupervised/` |

---

## Experiment 1a — Surface Agent Training Feasibility

**Research question:** Can the surface agent build a non-trivial object model from a
single sensorimotor exploration episode under real ToF sensor noise?

**Prereqs:** Lite 6 + Maixsense A010 wired and homed; parent config
`real_world/lite6_maixsense_unsupervised` operational (no further sensor calibration
required — see [feedback note on Lite 6 hand-eye extrinsics](#config-inheritance)).

**Objects:** `washbag`, `cardboard_box`, `mc_fox`, `cap` (4 of the 8 dissertation
objects).

**Orientations:** ORI0 only.

### Protocol (already executed — reproduction reference)

Each run trained for one episode with `run_name` overridden per object/repeat.

```bash
# General form — re-run only if you intend to extend the surface dataset.
python run.py experiment=real_world/lite6_maixsense_unsupervised \
  experiment.config.logging.run_name=real_world_lite6_maixsense_unsupervised_<TAG>
```

`<TAG>` values used:

| Object | Tags |
|---|---|
| washbag | `washbag1`, `washbag2`, `washbag3` |
| cardboard_box | `CBOX1`, `CBOX2`, `CBOX3` |
| mc_fox | `MCFOX` |
| cap | `CAP` |

### Outputs

For each `<TAG>`:

- `experiment_results/real_world_lite6_maixsense_unsupervised_<TAG>/0/model.pt` — learned graph checkpoint.
- `experiment_results/real_world_lite6_maixsense_unsupervised_<TAG>/train_stats.csv` — single row per run, ~36 monty steps, `result=unknown_object_not_matched_(TN)` (expected — unsupervised).
- `experiment_results/real_world_lite6_maixsense_unsupervised_<TAG>/log.txt` — full motion + frame timing log.

### Metrics reported

Qualitative: presence of a non-trivial graph. Quantitative metrics (point count,
bbox extent, mean inter-point distance, episode time) are produced by the
[Limitations chapter](#limitations--surface-agent-sensor-noise) analysis.

### Pass criteria / expected behaviour

- `model.pt` loads successfully via `torch.load` and contains a graph with > 0 nodes.
- `train_stats.csv` has exactly one row with `result == unknown_object_not_matched_(TN)`.
- No surface eval is performed and no `eval_stats.csv` is produced.

---

## Experiment 1b — Distant Agent Training (COMPLETE)

**Research question:** Can Monty learn the 7 dissertation objects from a single ZED
RGBD scene per object? (This is a supervised learning experiment that creates independent pretrained object models that don't accidentally merge like they might in a continuous learning experiment)

**Prereqs:** ZED 2i connected; pre-captured ORI0 RGBD scenes available to the
manual scene picker.

**Objects:** O1–O7 (`tbp_mug`, `sw_mug`, `tea_tin`, `mc_fox`, `hexagons`, `cap`, `washbag`).

**Orientations:** ORI0 only.

### Protocol

1. Place each object at the canonical mark in turn (or use the pre-captured ORI0 scene).
2. Run:
   ```bash
   python run.py experiment=real_world/dissertation/exp1_distant_train
   ```
3. At each scene-picker prompt, type `capture` to record a fresh frame, or select the
   pre-captured ORI0 frame for the current object.
4. Note the resulting checkpoint path.

### Outputs

- `experiment_results/exp1_distant_train/pretrained/0/model.pt` — supervised distant model.
- `experiment_results/exp1_distant_train/train_stats.csv` — one row per object.

### Metrics reported

Training does not report accuracy. Confirmation only: 7 graphs exist in `model.pt`.

### Pass criteria

- Checkpoint path saved; pass it to all downstream distant experiments via
  `experiment.config.model_name_or_path`.

---

## Experiment 1c — Distant Agent Eval (ORI0 baseline) (COMPLETE)

**Research question:** Can Monty recognise the 7 objects at canonical orientation?

**Prereqs:** Exp 1b checkpoint.

**Objects:** O1–O7.

**Orientations:** ORI0.

### Protocol

```bash
python run.py experiment=real_world/dissertation/exp1_distant_eval \
  experiment.config.model_name_or_path=experiment_results/exp1_distant_train/pretrained/0/model.pt
```

Run ≥3 episodes per object using fresh captures or the pre-captured ORI0 scenes.

### Outputs

- `experiment_results/exp1_distant_eval/eval_stats.csv` — one row per episode.

### Metrics reported

`percent_correct`, `percent_confused`, `percent_no_match`, `avg_rotation_error`,
`avg_num_lm_steps`, `avg_num_monty_matching_steps`.

### Pass criteria

- ≥3 episodes per object recorded.
- Aggregate `percent_correct` reported per object and overall.

---

## Experiment 1d — TBP Pretrained Sanity (mug only, optional)

**Research question:** Does TBP's stock `surf_agent_1lm_numenta_lab_obj` model
generalise to your physical mug as a sim-to-real baseline?

**Prereqs:** TBP pretrained model already on disk (loaded by the stock
`zed_supervised_eval` config).

**Objects:** O1 `tbp_mug` only.

**Orientations:** ORI0.

### Protocol

```bash
python run.py experiment=real_world/zed_supervised_eval
```

### Outputs

- `experiment_results/zed_supervised_eval/eval_stats.csv`.

### Metrics reported

`percent_correct` against the TBP sim-trained mug model.

### Pass criteria

- Reported as-is; any non-zero recognition is a positive sim-to-real transfer signal.

---

## Experiment 2 — Distant Rotation Invariance

**Research question:** Does recognition accuracy hold when objects are presented in
orientations not seen during training?

**Prereqs:** Exp 1b checkpoint; ORI1–ORI4 RGBD scenes already captured per object on
the fixed table mark.

**Objects:** O1–O7.

**Orientations:** ORI1, ORI2, ORI3, ORI4.

### Protocol

Run the eval config once per orientation, overriding `run_name`.

```bash
# ORI1 — 90° yaw
python run.py experiment=real_world/dissertation/exp2_distant_eval_rot \
  experiment.config.model_name_or_path=experiment_results/exp1_distant_train/pretrained/0/model.pt \
  experiment.config.logging.run_name=exp2_distant_eval_rot1

# ORI2 — 180° yaw
python run.py experiment=real_world/dissertation/exp2_distant_eval_rot \
  experiment.config.model_name_or_path=experiment_results/exp1_distant_train/pretrained/0/model.pt \
  experiment.config.logging.run_name=exp2_distant_eval_rot2

# ORI3 — 270° yaw
python run.py experiment=real_world/dissertation/exp2_distant_eval_rot \
  experiment.config.model_name_or_path=experiment_results/exp1_distant_train/pretrained/0/model.pt \
  experiment.config.logging.run_name=exp2_distant_eval_rot3

# ORI4 — off-axis tilt
python run.py experiment=real_world/dissertation/exp2_distant_eval_rot \
  experiment.config.model_name_or_path=experiment_results/exp1_distant_train/pretrained/0/model.pt \
  experiment.config.logging.run_name=exp2_distant_eval_rot4
```

### Outputs

- `experiment_results/exp2_distant_eval_rot{1,2,3,4}/eval_stats.csv`.

### Metrics reported

Compare `percent_correct` and `avg_rotation_error` across ORI0 (Exp 1c) → ORI4.

### Pass criteria

- All 4 orientations completed with ≥3 episodes per object.
- Pose-invariant if accuracy stays high and `avg_rotation_error` stays low at unseen
  orientations.

---

## Experiment 3 — Modality Discussion (analysis-only)

**Research question:** Which modality (ToF vs. RGBD) works better for which type of
object, and why?

**Prereqs:** Exp 1c CSV; Limitations-chapter graph stats (see below).

**Objects:** O1–O7 (distant), plus surface qualitative notes for O4, O6, O7.

**Orientations:** ORI0.

### Protocol

No new hardware run. Run the analysis pipeline:

```bash
python -m tools.dissertation_analysis --experiments exp3
```

Aggregate `percent_correct` and `avg_num_lm_steps` per object from Exp 1c, classify
each object as ToF-favourable / ZED-favourable / Ambiguous, and pair the distant
numbers with surface graph-quality notes for the 3 shared objects.

### Outputs

- `benchmarks/dissertation/analysis/exp3/modality_comparison.md` — distant per-object
  table + qualitative surface paragraph.
- `benchmarks/dissertation/analysis/exp3/complementarity.png` — per-object scatter.

### Metrics reported

Distant per-object `percent_correct`, `avg_num_lm_steps`, episode duration; surface
graph-quality summary referenced from the Limitations chapter.

### Pass criteria

- Side-by-side per-object commentary produced for the 3 shared objects (`mc_fox`,
  `cap`, `washbag`).

---

## Experiment 4 — Distant Continual Learning

**Research question:** Can Monty learn new objects incrementally without labels,
while retaining models of previously seen objects?

**Prereqs:** None — fresh unsupervised run, no prior checkpoint.

**Objects:** O1-O2, O3-O6 (`tbp_mug`, `sw_mug`, `mc_fox`, `hexagons`, `cap`).

**Orientations:** ORI0.

### Protocol

3 epochs, each epoch contains 6 episodes. In the first epoch a new object is introduced each episode. The same object are then introduced across epochs 2 and 3 (no labels supplied):

```bash
python run.py experiment=real_world/dissertation/exp4_distant_continual
```

At each scene-picker prompt, select/capture the ZED scene that matches the episode
in the sequence above.

### Outputs

- `experiment_results/exp4_distant_continual/train_stats.csv`.
- `experiment_results/exp4_distant_continual/0/model.pt`.

### Metrics reported

```bash
python -c "
from tbp.monty.frameworks.utils.logging_utils import print_unsupervised_stats
print_unsupervised_stats('experiment_results/exp4_distant_continual/train_stats.csv')
"
```

- `mean_objects_per_graph` (target → 1).
- `mean_graphs_per_object` (target → 1).
- Episode-by-episode hit/miss for return visits (episodes 6–10).

### Pass criteria

- All 10 episodes captured; both metrics within 0.5 of the ideal 1.0.
- Return-visit episodes (6–10) classified correctly above chance.

---

## Experiment 5 — Cross-Agent Transfer (discussion only)

**Research question:** Are Monty's object models sensor-agnostic — can a model built
with one sensor be recognised through the other?

**Status:** Not executed. No surface eval data exists, and no further surface runs
will be performed.

**Discussion:** The architectural plumbing for cross-agent transfer is in place — the
8 surface `model.pt` checkpoints (Exp 1a) load with the same Monty configuration used
by the distant agent, and the distant `model.pt` is in the same format. What is
missing is the eval-time data to populate the off-diagonal cells of the transfer
matrix:

| | Eval: Surface Agent | Eval: Distant Agent |
|---|---|---|
| **Train: Surface Agent** | not executed | not executed |
| **Train: Distant Agent** | not executed | Exp 1c (baseline) |

Surface agent recognition was not reliable enough under sensor noise (see Limitations
chapter) to support quantitative cross-agent claims. The dissertation discussion notes
that `EvidenceGraphLM` does not encode HSV features in the surface model (the surface
sensor module exposes `hsv` as a feature, but the surface-trained graphs in this
dataset were captured under unsupervised settings without the colour-rich features
that the distant model relies on for discrimination). This is documented as a known
limitation with engineering implications for future work.

---

## Experiment 6 — Distant Similar Object Discrimination

**Research question:** Can the distant agent distinguish geometrically similar
objects?

**Prereqs:** Exp 1b checkpoint.

**Objects:** O1 `tbp_mug` and O3 `tea_tin` (the chosen similar pair).

**Orientations:** ORI0.

### Protocol

```bash
python run.py experiment=real_world/dissertation/exp6_distant_similar_eval \
  experiment.config.model_name_or_path=experiment_results/exp1_distant_train/pretrained/0/model.pt
```

3 epochs, each with 2 episodes. Both `tbp_mug` and `tea_tin` are introduced once per epoch.

### Outputs

- `experiment_results/exp6_distant_similar_eval/eval_stats.csv`.

### Metrics reported

`percent_correct`, `percent_confused`, and the 2×2 confusion matrix (`tbp_mug` ↔
`tea_tin`).

### Pass criteria

- Confusion matrix populated for both objects.
- Modality interpretation: HSV features should help the distant agent discriminate
  surface-texture/colour differences between the two mugs.

---

## Limitations — Surface Agent Sensor Noise

**Research question:** How repeatable is the surface-agent learned graph under fixed
object pose and a noisy ToF sensor?

**Prereqs:** Exp 1a outputs (8 unsupervised training run dirs).

**Objects:** `washbag` (3 runs), `cardboard_box` (3 runs); `mc_fox` and `cap`
contribute single-run reference graphs.

**Orientations:** ORI0.

### Protocol

Analysis-only — no hardware run.

```bash
python -m tools.dissertation_analysis --experiments surface_unsupervised
```

For `washbag` and `cardboard_box`, the analysis compares the 3 repeat runs across:

- learned graph point count
- graph bounding-box extent (xyz, mm)
- mean inter-point distance
- episode wall-clock time

### Outputs

- `benchmarks/dissertation/analysis/surface_unsupervised/learned_graphs/<run_tag>.png`
  — one 3D `plot_graph()` figure per run.
- `benchmarks/dissertation/analysis/surface_unsupervised/summary.md` — per-run table.
- `benchmarks/dissertation/analysis/surface_unsupervised/repeatability.md` —
  washbag×3 and CBOX×3 comparison with `(max − min)` delta column.
- `benchmarks/dissertation/analysis/surface_unsupervised/repeatability.png` — bar
  chart of point count and bbox volume across the three repeats per object.

### Metrics reported

Per-run graph point count, bbox extent, mean inter-point distance, episode time;
plus the variance summary across repeats.

### Pass criteria

- All 8 runs render a learned-graph figure.
- Repeatability table populated for `washbag` and `cardboard_box`; the `(max − min)`
  column is reported and discussed as the dominant sensor-noise signal.

For the architectural background to this section, see
[`docs/operational-guides/dissertation-surface-agent-chapter.md`](../../../../../../docs/operational-guides/dissertation-surface-agent-chapter.md).

---

## Config inheritance

| Config | Parent |
|---|---|
| `exp1_distant_train`, `exp4_distant_continual` | `/experiment/real_world/zed_continual_manual_train` |
| `exp1_distant_eval`, `exp2_distant_eval_rot`, `exp6_distant_similar_eval` | `/experiment/real_world/zed_supervised_eval` |
| Surface unsupervised runs (Exp 1a) | `/experiment/real_world/lite6_maixsense_unsupervised` (parent used directly) |

These parents encode tuned hardware defaults (12 cm working distance, depth-burst
averaging, goal dispatch, semantic bounds filtering, validated Lite 6 hand-eye
extrinsics). Do not duplicate them.

To inspect the fully-resolved config for any experiment:

```bash
python run.py experiment=real_world/dissertation/exp1_distant_train --cfg job
```

Commit the output to `benchmarks/dissertation/resolved_configs/` for reproducibility.

---

## Output structure

Distant runs:

```
experiment_results/exp1_distant_train/
  0/
    model.pt
    exp_state_dict.pt
    config.pt
  train_stats.csv
  eval_stats.csv     # if do_eval: true
```

Surface unsupervised runs (Exp 1a):

```
experiment_results/real_world_lite6_maixsense_unsupervised_<TAG>/
  0/
    model.pt
    exp_state_dict.pt
    config.pt
  train_stats.csv    # 1 row, unknown_object_not_matched_(TN)
  log.txt            # full motion + frame timing log
```

Per-experiment analysis outputs aggregate under
`benchmarks/dissertation/analysis/<experiment>/`.

---

## Recommended run order

1. `exp1_distant_train`
2. `exp1_distant_eval`
3. `zed_supervised_eval` (optional sanity check, mug only)
4. `exp2_distant_eval_rot1..4` (one run per orientation)
5. `exp4_distant_continual`
6. `exp6_distant_similar_eval`
7. Run the analysis pipeline:
   ```bash
   python -m tools.dissertation_analysis --experiments all
   ```
   Generates Exp 3 modality discussion, Exp 5 narrative stub, and the surface
   limitations chapter outputs.

The surface unsupervised training (Exp 1a) is already complete and requires no
further hardware execution.
