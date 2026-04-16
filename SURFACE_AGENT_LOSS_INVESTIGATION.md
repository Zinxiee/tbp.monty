# Surface Agent Object Loss Investigation

**Status**: Root cause identified, implementation pending  
**Date**: April 2026  
**Target**: Fix surface agent losing object within first ~12 steps of real-world probe episodes

---

## Executive Summary

Surface agent (Lite6 robot + Maixsense ToF sensor) loses tracked object during probe acquisition within steps 5–15. Investigation eliminated motion/timing/compensation as root causes. **Primary culprit: fallback + recovery policy interaction.**

When center of semantic mask goes off-object but coverage remains, fallback state extracts features from periphery (often invalid), blocks valid learning module (LM) input via `use_state` gate, and policy acts on weak signals without recovery pathway. Result: object coverage collapses to zero and agent cannot re-acquire.

---

## Problem Statement

### Observed Behavior
- Real-world probe episodes start normally (settles on object, observes features).
- Within ~12 steps, semantic mask center shifts off-object.
- Agent cannot recover; coverage drops to zero.
- Episode terminates prematurely with 10–20% fewer observations than simulated baseline.

### Hypothesis Evolution
1. **Settle gate convergence** (false accept on near-miss poses) → **Rejected**: 3mm + 2-sample hardening did not prevent loss.
2. **Motion-to-observation lag** (action ↔ observation timing mismatch) → **Rejected**: Settle logs show ~1–2mm final error, motion execution clean.
3. **Orientation compensation** (rotation → compensating translation direction/magnitude) → **Rejected**: Rotation-only mode still lost object.
4. **Semantic bounds filtering** (world-frame Y/X/Z gates too restrictive) → **Partially relevant**: Y filter (table rejection) is correct, X/Z are workspace boundaries; not primary cause of loss.
5. **Fallback + recovery policy** (center loss state produces invalid morphology, blocks LM input, weak recovery) → **Accepted**: Matches all observations, needs further evidence.

---

## Root Cause Analysis

### Fallback Activation Sequence
```
Step N:   Center on-object, coverage=100, LM receives valid morphology
Step N+1: Motion execution, center shifts near edge or off-object
Step N+2: Semantic processing: center_id off mask, but coverage=20–50 pixels remain
          → Fallback condition met: on_object=False && is_surface_sm=True && coverage > 0
          → Extract features from periphery (invalid normals, weak curvature)
Step N+3: Policy receives invalid morphology (default [0,0,1] normal)
          → LM input blocked (use_state requires center_on_object=True)
          → Policy acts blindly on invalid sensory signal
Step N+4: Agent moves further off-object
Step N+5: Coverage collapses to zero
          → Recovery mode too late (should have recentered at Step N+2)
```

### Why This Matters
- **LM starvation**: When fallback activates, `use_state` gate closes. LM stops learning exactly when recovery is needed.
- **Invalid morphology propagation**: Fallback extracts from table-contaminated pixels and periphery, producing normals that don't match object surface.
- **No recenter pathway**: Policy has no forced recenter action when center is lost; instead blindly trusts invalid signals.
- **Search mode too late**: Recovery/search mode only engages after coverage → 0, missing the window when center can still be found.

### Evidence
**Bounds isolation probe configs revealed**:
- `bounds_off` (all world filters disabled): Center stays on-object (false positive), agent still moves off. → Proves masking not root cause.
- `bounds_y_only` (only table rejection): Center correctly flips off-object, but fallback + weak policy caused loss. → Proves fallback state exists and is problematic.
- `bounds_x_only`, `bounds_z_only`: X/Z workspace limits are mostly benign; loss independent of their state.

---

## Technical Implementation Details

### Hardware Setup
- **Robot**: UFactory Lite6
- **Sensor**: Maixsense A010 ToF (100×100 depth, 10×10 patch ROI, 70°H × 60°V FOV)
- **Camera**: Zed 2i stereo (distant agent, not involved in this loss)
- **Object size**: upside down bowl/cube (~100mm)

### Key Architecture Components

#### 1. **Semantic Filtering Pipeline** (`multimodal_monty_meets_world/maixsense_a010_api/monty_adapter.py`)
```
Raw depth → Validity mask → World-frame transform → Bounds filter
                                                     (y_min, x_min/max, z_min/max)
                                                     → Semantic_3d output
```
- **world_y_min_m = 0.01m**: Rejects table surface (working correctly).
- **x_min/max = ±0.3m**: Workspace bounds (mostly benign for this loss).
- **z_min/max**: Workspace bounds (mostly benign for this loss).
- **Diagnostic**: Added `SEMANTIC_FILTER_COUNTS` event (tracks depth_valid → normalized → post_world counts).

#### 2. **Sensor Module Gating** (`src/tbp/monty/frameworks/models/sensor_modules.py`)
```python
# Line ~258: SM gating logic
if not center_on_object:
    if is_surface_sm and object_coverage > 0:
        # FALLBACK: Try periphery
        attempt morphology extraction from off-center pixels
    else:
        # NO FALLBACK: Skip feature extraction
        return default signals

# Line ~325: use_state gate (LM input eligibility)
use_state = on_object and valid_signals
# If False, LM does not receive morphology (learning blocked)
```

#### 3. **Surface Policy** (`multimodal_monty_meets_world/real_world_surface_policy.py`)
- Orientation compensation: Angle clamped to ±15° to control translation magnitude.
- Decomposition: Returns (rotated_normal, branch, raw_angle) for diagnostics.
- Diagnostic toggles:
  - `disable_orient_compensation_translation`: Rotation-only mode.
  - `enable_orient_decomposition_logging`: Per-step normal decomposition.

#### 4. **Goal Settling** (`src/tbp/monty/frameworks/environments/real_world_environment.py`)
- **Default tolerance**: 3mm position, 1° rotation.
- **Gate requirement**: 2 consecutive in-tolerance samples.
- **Hard-stop timeout**: Prevent infinite waiting.
- Status: Hardened this session; settles to ~1–2mm final error.

### Configuration Hierarchy
```
conf/environment/real_world_lite6_maixsense.yaml (base)
  └─ experiment/real_world/lite6_maixsense_unsupervised_probe_*.yaml
```

**Base environment config settings**:
```yaml
observation_adapter:
  semantic_debug_logging: false  # Added this session
  world_y_min_m: 0.01             # Table rejection (working)
  world_x_min_m: -0.3
  world_x_max_m: 0.3
  world_z_min_m: -0.5
  world_z_max_m: -0.1
  
settle_parameters:
  convergence_position_tolerance_mm: 3.0
  required_consecutive_samples: 2
```

---

## Diagnostic Infrastructure Built

### 1. Orientation Decomposition Logging
- **What**: Breaks normal into rotated component + static component.
- **Why**: Reveals whether policy is reacting to valid surface normals or fallback defaults.
- **Output**: `orient_{horizontal,vertical}_decomposition` log with angle breakdown.
- **Toggle**: `enable_orient_decomposition_logging` in real_world_surface_policy.py.

### 2. Semantic Filter Counts
- **What**: Tracks per-frame semantic pixel counts through filtering pipeline.
- **Stages**: depth_valid → normalized → post_bottom_patch → pre_world_bounds → post_world_bounds.
- **Why**: Reveals where object coverage collapses (bounds filter vs. other causes).
- **Output**: `SEMANTIC_FILTER_COUNTS` event (JSON) with all counts.
- **Toggle**: `semantic_debug_logging: true` in config.

### 3. Settle Convergence Monitoring
- **What**: Tracks pose tolerance and cumulative in-tolerance sample count.
- **Output**: `SETTLE_POSE_TARGET_VS_SENSED`, `settle_convergence_*` logs.
- **Verdict**: Convergence is working correctly (~1–2mm final error).

### 4. Bounds Isolation Probe Matrix
6 probe configs for incremental hypothesis testing:

| Config | Y Filter | X Filter | Z Filter | Translation | Notes |
|--------|----------|----------|----------|-------------|-------|
| `diagnostics_control` | ✓ | ✓ | ✓ | ✓ | Baseline + all diagnostics |
| `rotation_only` | ✓ | ✓ | ✓ | ✗ | Disable translation only |
| `bounds_off` | ✗ | ✗ | ✗ | ✓ | All world bounds null |
| `bounds_y_only` | ✓ | ✗ | ✗ | ✓ | Table rejection only |
| `bounds_x_only` | ✗ | ✓ | ✗ | ✓ | X workspace only |
| `bounds_z_only` | ✗ | ✗ | ✓ | ✓ | Z workspace only |

---

## Key Findings

### What is Working
✅ **Motion execution**: Settle converges to 1–2mm error, command pipeline clean.  
✅ **Table rejection**: Y filter correctly removes table pixels; absence causes false on-object states.  
✅ **World bounds**: X/Z workspace limits are mostly benign; loss independent of their state.  
✅ **Settle gate hardening**: 2-sample requirement prevents false convergence acceptance.  

### What is Failing
❌ **Fallback morphology quality**: Extracted features from off-center / table-contaminated pixels produce invalid normals.  
❌ **LM starvation**: use_state gate blocks learning input exactly when recovery policy needs it most.  
❌ **Recovery pathway**: No forced recenter action when center is lost; policy blindly trusts invalid signals.  
❌ **Policy delay**: Touch/search recovery starts only after coverage → 0, missing the recenter window.  

---

## Proposed Solution: Gated Fallback-Recovery Mode

### Implementation Scope

#### A. Modify `src/tbp/monty/frameworks/models/sensor_modules.py`

**1. Add recovery flag to sensor module state**
```python
# Around line 150 (in __init__)
self.center_lost_recovery_active = False
self.center_loss_duration_steps = 0
```

**2. Fallback condition check (line ~270)**
```python
if not center_on_object and is_surface_sm:
    if object_coverage > self.surface_fallback_coverage_threshold:
        # Check if morphology is valid
        invalid_morphology = (
            np.allclose(surface_normal, [0, 0, 1]) or  # default fallback normal
            curvature_distance < -0.001  # bad curvature
        )
        if invalid_morphology:
            self.center_lost_recovery_active = True
            self.center_loss_duration_steps += 1
            # Set flag: do not propagate invalid morphology
            valid_signals = False  # Block LM input
            reason_code = "center_lost_recovery_active"
        else:
            # Morphology valid, allow propagation
            valid_signals = True
    else:
        self.center_lost_recovery_active = False
        self.center_loss_duration_steps = 0
```

**3. Reset recovery flag when center returns**
```python
if center_on_object:
    self.center_lost_recovery_active = False
    self.center_loss_duration_steps = 0
```

#### B. Modify `multimodal_monty_meets_world/real_world_surface_policy.py`

**1. Add recovery action gating**
```python
def orient(self, observation):
    # Check recovery flag
    if self.sensor_state.center_lost_recovery_active:
        # Force recenter-only action (zero rotation, small touch)
        return {'angle': 0.0, 'axis': None, ...}
    
    # Normal orient logic (only if center is on-object)
    ...
```

**2. Block large compensating translations during recovery**
```python
def _compensating_distances(self, angle_rad):
    if self.sensor_state.center_lost_recovery_active:
        # Return zero translation to avoid moving further off
        return (0.0, 0.0)
    
    # Normal compensation logic
    ...
```

#### C. Add Config Parameters

Add to `conf/environment/real_world_lite6_maixsense.yaml`:
```yaml
sensor_module:
  surface_fallback_invalid_morphology_threshold: 0.001  # curvature validity check
  center_loss_recovery_timeout_steps: 5  # max steps before search mode
```

---

## Testing Plan

### Unit Tests
- Fallback flag activation/deactivation on center loss/recovery.
- Recovery action gating prevents large translations.
- LM input blocked during recovery, unblocked on center return.

### Integration Test (Probe Config)
```bash
python run.py experiment=real_world/lite6_maixsense_unsupervised_probe_diagnostics_control
```
**Expected outcome**:
- Step N: Center on-object, normal operation.
- Step N+1: Center goes off, fallback activates.
- Step N+2: Recovery flag on, policy forces recenter (zero rotation, small touch).
- Step N+3: Center back on-object, recovery completes, normal operation resumes.
- LM input resumes (use_state gate reopens).

### Success Metrics
- No object loss in first 50 steps (improvement from current ~12 steps).
- Fallback state duration < 2–3 steps (quick recovery).
- LM learning resumes after recovery (morphology signal returns).

---

## Current Status

### Completed
✅ Settle convergence hardening (3mm + 2-sample + hard-stop timeout).  
✅ Diagnostic infrastructure (rotation-only toggle, decomposition logging, semantic counts).  
✅ Factory wiring + config exposure (semantic_debug_logging parameter).  
✅ Unit test coverage (44/44 tests passing).  
✅ Bounds isolation probe matrix (4 configs created + tested).  
✅ Root cause identification (fallback + recovery policy interaction).  

### Files Modified This Session
1. `multimodal_monty_meets_world/real_world_surface_policy.py`: Diagnostic toggles, decomposition helper.
2. `multimodal_monty_meets_world/maixsense_a010_api/monty_adapter.py`: Semantic count tracking.
3. `multimodal_monty_meets_world/factory.py`: Passthrough semantic_debug_logging parameter.
4. `src/tbp/monty/conf/environment/real_world_lite6_maixsense.yaml`: Added semantic_debug_logging config.
5. Test files: Updated all unit tests for new toggles + parameters (44/44 passing).
6. Probe configs: Created 6 new experiment configs for hypothesis isolation.

### Pending
⏳ **Implement gated fallback-recovery mode** (sensor_modules.py + real_world_surface_policy.py).  
⏳ **Validate with diagnostics_control probe run**.  
⏳ **Compare recovery behavior against bounds_y_only baseline**.  

---

## Code References

### Key Files
- **Semantic filtering**: `multimodal_monty_meets_world/maixsense_a010_api/monty_adapter.py` (~200 lines)
- **Sensor module gating**: `src/tbp/monty/frameworks/models/sensor_modules.py` (lines 250–330, 380–420)
- **Surface policy**: `multimodal_monty_meets_world/real_world_surface_policy.py` (orient/touch + compensation)
- **Config base**: `src/tbp/monty/conf/environment/real_world_lite6_maixsense.yaml`
- **Experiment configs**: `src/tbp/monty/conf/experiment/real_world/lite6_maixsense_unsupervised_probe_*.yaml`

### Key Log Events for Debugging
- `SEMANTIC_FILTER_COUNTS`: Tracks pixel counts through filter pipeline.
- `SM gating`: Prints on_object, coverage, reason_code for each step.
- `orient_*_decomposition`: Normal angle breakdown (rotated vs. static).
- `SETTLE_POSE_TARGET_VS_SENSED`: Settle convergence traces.

---

## Tools Used

### Commands for Reproduction
```bash
# Run single probe config
python run.py experiment=real_world/lite6_maixsense_unsupervised_probe_diagnostics_control

# Run all unit tests
pytest

# Run single test file
pytest tests/unit/frameworks/test_real_world_lite6_probe_configs.py

# Check for lint/type errors
ruff check
mypy src/
```

### Diagnostic Analysis
```bash
# Summarize loss events from log
python tools/real_world_surface_loss_summary.py --log-file <path>
```

---

## Next Steps

1. **Implement gated fallback-recovery mode** (sensor_modules.py + real_world_surface_policy.py).
2. **Run diagnostics_control probe** with new gating to verify recovery behavior.
3. **Compare against bounds_y_only baseline** to confirm improvement.
4. **Iterate on recovery timeout + threshold parameters** based on results.
5. **(Future)**: Consider adaptive table threshold (dy_min) per object size.

---

## Questions for Next Chat

- Should recovery timeout be fixed (5 steps) or adaptive (based on coverage trajectory)?
- Should policy preserve LM input during recovery (with lower confidence) or block it completely?
- Any known failure modes in fallback morphology extraction that should be explicitly guarded against?
- Should recovery actions be "touch in place" (zero translation) or "gentle recenter search"?

---

## Appendix: Log Example (bounds_y_only failure)

```
Step 12: SM gating: center_id=98, semantic_id=1.0, on_object=True, coverage=0.95
         → LM input active, learning normal [0.2, 0.1, 0.96]

Step 13: Motion executed (orient + touch)
         → Agent moved toward surface, coverage dipped

Step 14: SM gating: center_id=55, semantic_id=0.0, on_object=False, coverage=0.20
         → FALLBACK activates, morphology extracted from periphery (invalid)
         → LM input BLOCKED (use_state=False)
         → Policy receives invalid normal [0, 0, 1]

Step 15: Policy acts on invalid signal
         → Moves agent further off-object (wrong direction)

Step 16: SM gating: center_id=off, semantic_id=0.0, on_object=False, coverage=0.05
         → Coverage collapsed

Step 17: Coverage → 0, recovery mode too late
         → Episode terminates
```

**With gated fallback-recovery fix**:
```
Step 14: SAME: center off, coverage=0.20
         → NEW: Recovery flag set, LM input blocked, policy forced to recenter
         → Policy action: zero rotation, small touch inward

Step 15: Agent recentered, coverage → 0.50, center → on-object
         → Recovery flag cleared, LM input UNBLOCKED
         → Normal learning resumes

Step 16+: Successful object tracking continues
```
