# Surface Agent Loss Findings Update

**Date**: 2026-04-16  
**Context**: Follow-up to `SURFACE_AGENT_LOSS_INVESTIGATION.md` using new per-axis semantic rejection logging.

## TL;DR
- New diagnostics show object-loss events are dominated by **world Y-min rejection** (table/low-height filtering), not X/Z bounds.
- Critical failure sequence observed:
  1. Agent still on-object with valid morphology.
  2. Large `OrientVertical` correction (`raw≈66°`, clamped to `15°`) plus compensating translation (`~3.0 cm` norm).
  3. Next frame: `SEMANTIC_FILTER_COUNTS post_world=0`, `rejects={y_min:100,...}`.
- Interpretation: motion places full patch into low-Y region; semantic mask collapses to zero; fallback path never activates because coverage is already zero.

## New Evidence
From latest `diagnostics_control` run:
- Before loss:
  - `object_coverage=0.76`
  - `post_world=76`
  - `rejects={y_min:24,x_min:0,x_max:0,z_min:0,z_max:0,any:24}`
- Loss-triggering step:
  - Policy logs: `orient_vertical raw=66.129°, shaped=15.000°`
  - Environment logs: `RELATIVE_ACTION_GOAL ... delta_norm_m=0.030299` (no clipping)
- Immediately after:
  - `SEMANTIC_FILTER_COUNTS ... post_world=0 ... rejects={y_min:100,x_min:0,x_max:0,z_min:0,z_max:0,any:100}`
  - SM gate flips to `on_object=False`, `object_coverage=0.0`, `should_extract=False`

## Updated Root-Cause Ranking
1. **Primary**: Recovery/search-orient motion amplitude pushes sensed patch into low-Y region, causing all pixels to fail `world_y_min_m`.
2. **Secondary**: Once coverage becomes zero, no fallback morphology can be extracted; LM input remains off.
3. **Lower**: X/Z world bounds are not active contributors in this observed failure slice.
4. **Lower**: Settle/timing issues remain unlikely (convergence gaps ~1–2 mm).

## What This Changes
- Prior hypothesis (“fallback invalid morphology as first trigger”) is now likely **episodic**, not universal.
- In this run, the first hard break is **semantic collapse to zero caused by Y-bound after aggressive orient step**.

## Recommended Next Runs (Small Matrix)
1. `diagnostics_control` with reduced orient aggressiveness (e.g., max orient 8–10°).
2. `rotation_only` (translation disabled) with per-axis reject logging enabled.
3. Optional calibration probe: temporary `world_y_min_m` relaxation (small delta only) to confirm sensitivity.

Compare:
- first step where `rejects.y_min` spikes,
- first `post_world=0` step,
- steps survived before unrecoverable loss.

## Immediate Implementation Direction
- Add **near-loss limiter** in `RealWorldSurfacePolicy`:
  - when coverage drops below threshold, reduce orient angle and/or compensating translation.
- Keep current semantic Y-min filter in place (still correct for table rejection), but prevent policy from driving into all-table view.
