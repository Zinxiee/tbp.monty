# Maixsense A010 + UFactory Lite6 Real-World Bring-Up Runbook

**Version:** 1.0  
**Target Hardware:** UFactory Lite6 + Maixsense A010 ToF  
**Operating Mode:** USB-direct streaming, strict blocking control  
**Objective:** Unsupervised object learning with manual object workflow

---

## Quick Start Checklist

- [ ] Lite6 IP reachable and motion-enabled
- [ ] A010 USB port discovered and accessible
- [ ] Robot at known safe home pose (300, 0, 250) mm
- [ ] Sensor intrinsics calibrated (static or HTTP fetch)
- [ ] First dry-run (motor commands off) completes without halt
- [ ] Object in workspace, operator present for first manual episodes

---

## 1. Hardware Setup & Discovery

### UFactory Lite6 Network Access

```bash
# Verify robot IP reachable
ping 192.168.1.159

# Expected: ICMP responses, round-trip < 50ms typical
```

**If unreachable:**
- Check network cable / WiFi connection
- Verify IP address matches your deployment (default: 192.168.1.159)
- Use xArm web interface to confirm robot is online

### Maixsense A010 USB Port Discovery

```bash
# List all USB serial ports
python -c "import serial.tools.list_ports; [print(f'{p.device}: {p.description}') for p in serial.tools.list_ports.comports()]"

# Expected output (example):
# /dev/ttyUSB0: USB2.0-Serial
# /dev/sipeed: Sipeed USB Device (symlink, optional)

# Or, create stable udev symlink:
sudo vi /etc/udev/rules.d/99-maixsense.rules
# Add: SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", SYMLINK+="sipeed", MODE="0666"
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/sipeed  # Should exist after trigger
```

**If no USB devices found:**
- Check USB cable (must be USB 2.0 or higher for baud 921600)
- Verify sensor power (LED on Maixsense board should be lit)
- Try different USB port (USB 3.0 may cause timing issues for some boards)

---

## 2. Sensor Calibration: Intrinsics

The Maixsense A010 requires pinhole camera intrinsics (focal lengths, principal point) for depth unprojection. Choose one approach:

### Option A: Fetch from HTTP (Recommended for Production)

If your A010 is connected to Ethernet and accessible via HTTP:

```python
from multimodal_monty_meets_world.maixsense_a010_api import (
    MaixsenseA010HTTP,
    create_adapter_from_http_calibration,
)

http = MaixsenseA010HTTP("192.168.233.1", 80)
adapter = create_adapter_from_http_calibration(http)

# Intrinsics now loaded; store or reference in config
print(f"fx={adapter.intrinsics.fx}, fy={adapter.intrinsics.fy}")
print(f"cx={adapter.intrinsics.cx}, cy={adapter.intrinsics.cy}")
```

Update your Hydra config:

```yaml
observation_adapter:
  _target_: multimodal_monty_meets_world.factory.create_observation_adapter
  fx: 232.5    # From HTTP fetch above
  fy: 231.8
  cx: 159.2
  cy: 119.7
```

### Option B: Use Known Calibration (Development)

If you have a pre-calibrated intrinsics file or known values:

```yaml
observation_adapter:
  _target_: multimodal_monty_meets_world.factory.create_observation_adapter
  fx: 230.0    # Placeholder safe values (works for most A010 units)
  fy: 230.0
  cx: 160.0
  cy: 120.0
```

**Verify:** Quick depth unprojection (3D point should land in workspace).

---

## 3. Robot Safety Configuration

The Lite6 adapter includes comprehensive safety checks. Customize workspace bounds and motion limits in the Hydra config:

```yaml
goal_adapter_config:
  workspace_min_xyz_m: [0.10, -0.35, 0.02]  # x, y, z min in meters
  workspace_max_xyz_m: [0.70, 0.35, 0.45]   # x, y, z max
  max_translation_step_m: 0.08               # Max move per step (meters)
  max_rotation_step_deg: 20.0                # Max rotation (degrees)
  convergence_timeout_s: 1.2                 # Wait for move completion
  convergence_position_tolerance_mm: 5.0     # Position reached tolerance
  payload_mass_kg: 0.056                     # Gripper + object estimate
```

**Verify workspace:** Place object at corners and edges; confirm safe reachability.

---

## 4. Home Pose Verification

Before first run, manually move the robot to your desired home pose and record joint angles:

```python
from multimodal_monty_meets_world.ufactory_api.robot_interface import RobotInterface
import time

robot = RobotInterface('192.168.1.159')
time.sleep(0.5)  # Wait for XArmAPI to stabilize
sense_data = robot.get_sense_state()
print("End-effector pose:", sense_data['end_effector'])
# Expected: [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
```

Update config with this pose:

```yaml
env_init_args:
  home_pose_mm_deg: [300.0, 0.0, 250.0, 180.0, 0.0, 0.0]
```

---

## 5. First Dry Run (Motor Commands Disabled)

Before moving hardware, validate the pipeline with commands sent to logging only:

```bash
cd /root/tbp

# Run a short unsupervised experiment with mocking
python -m pytest tests/integration/frameworks/environments/real_world_lite6_a010_smoke_test.py -v

# Expected: All tests pass, no hardware moves
```

If smoke test fails:
- Check import paths (multimodal_monty_meets_world must be in PYTHONPATH)
- Verify factory functions can instantiate objects
- Check Hydra config syntax (use `hydra.utils.instantiate(cfg.env_init_args)` to debug)

---

## 6. Single-Step Dry Run with Hardware (Polling Only)

Once smoke test passes, execute one controlled step on real hardware:

```bash
cd /root/tbp

# Create a temporary test script
cat > /tmp/test_single_step.py << 'EOF'
import sys
sys.path.insert(0, '/root/tbp')

from multimodal_monty_meets_world.factory import (
    create_robot_interface,
    create_usb_frame_client,
    create_observation_adapter,
    create_goal_adapter,
)
from tbp.monty.frameworks.environments.real_world_environment import RealWorldLite6A010Environment

# Create hardware objects
robot = create_robot_interface(ip_address='192.168.1.159', start_listener=True)
sensor = create_usb_frame_client(port='/dev/sipeed', baudrate=921600)
adapter = create_observation_adapter(fx=230.0, fy=230.0, cx=160.0, cy=120.0)
goal_adapter = create_goal_adapter(robot=robot)

# Create environment and reset
env = RealWorldLite6A010Environment(
    robot_interface=robot,
    sensor_client=sensor,
    observation_adapter=adapter,
    goal_adapter=goal_adapter,
    home_pose_mm_deg=[300.0, 0.0, 250.0, 180.0, 0.0, 0.0],
    require_object_swap_confirmation=True,
)

obs, prop = env.reset()
print(f"Reset complete. Proprioceptive state: {prop}")

# One step
obs, prop = env.step([])
print(f"Step 1 complete.")
print(f"Observation keys: {list(obs.keys())}")

env.close() if hasattr(env, 'close') else None
EOF

python /tmp/test_single_step.py
```

**Monitor:** Watch robot at home pose, wait for "Swap object..." prompt, press Enter. Sensor should stream frames without errors.

---

## 7. Emergency Stop & Safety Overrides

### Graceful Shutdown

If hardware behaves unexpectedly during a run:

```python
# In a separate terminal/REPL
robot = RobotInterface('192.168.1.159')
robot.arm.motion_enable(enable=False)
robot.stop_listening()
```

### Hard Stop Reason Codes (Logged)

The environment logs safety stops with explicit reason codes:

- `SENSOR_TIMEOUT`: No frames from A010 within timeout
- `ROBOT_API_UNHEALTHY`: Robot joint/position API returned error code
- `GOAL_DISPATCH_REJECTED`: Goal-pose safety check failed (workspace/step/rotation bounds)
- `UNSUPPORTED_ACTION`: Policy issued action type not supported by real-world env
- `INVALID_HOME_POSE`: Could not move to home on reset

Check logs for reason:

```bash
tail -f /root/tbp/outputs/*/logs/experiment.log | grep "RealWorldSafetyStopError\|SENSOR_TIMEOUT\|ROBOT_API"
```

### Override Intrinsics at Runtime

If sensor intrinsics need adjustment mid-run:

```yaml
# Edit environment config's observation_adapter section
observation_adapter:
  _target_: ...
  fx: 235.0    # Adjusted from 230.0
  fy: 235.0    # Adjusted
  cx: 161.0    # Adjusted
  cy: 120.5    # Adjusted
```

Re-run experiment with updated config.

---

## 8. Troubleshooting Common Issues

### "No frame received from Maixsense A010"

**Diagnosis:**
- Check USB connection: `lsusb | grep -i maixsense` or `ls -l /dev/sipeed`
- Verify baud rate: Device expects 921600 (not 115200)
- Check frame timeout: Sensor may be slow to boot

**Fix:**
```bash
# Restart USB connection
sudo sh -c 'echo 0 > /sys/bus/usb/devices/X/power/autosuspend_delay_ms'  # Disable auto-suspend if needed

# Increase timeout in config
sensor_client:
  _target_: ...
  timeout: 0.1  # Increase from 0.05
```

### "Robot IP unreachable"

**Diagnosis:**
- Network misconfiguration or robot powered down
- Firewall blocking motion port (default: 502 for xArm)

**Fix:**
- Power cycle robot; wait 30 seconds for boot
- Verify IP with `arp-scan` or robot web UI
- Update config with correct IP

### "Goal dispatch rejected: workspace bounds"

**Diagnosis:**
- Policy is commanding goal pose outside defined workspace

**Fix:**
- Widen workspace bounds in config (if object is in that region)
- Verify home pose and object placement are within bounds
- Inspect goal pose being dispatched: add debug logging to goal_adapter

### "Listener thread not started"

**Diagnosis:**
- Robot initialization failed silently

**Fix:**
```python
robot = create_robot_interface(..., start_listener=True)
assert robot.is_listening == True  # Verify thread started
```

---

## 9. First Unsupervised Learning Run

Once dry run and single step pass:

```bash
cd /root/tbp

python -m tbp.monty.frameworks.run \
  --config-path conf \
  --config-name experiment/real_world/lite6_maixsense_unsupervised \
  experiment.config.env_interface=real_world_manual_train_eval \
  experiment.num_train_steps=5 \
  experiment.num_eval_steps=2
```

**Monitor:**
- First 5 training steps: motor commands → goal dispatch → settle → sense
- 2 eval steps: same flow, different policy mode
- Terminal prompts for object swap between train/eval

**Success Criteria:**
- No `RealWorldSafetyStopError`
- Observations logged (depth, semantics, proprioceptive state)
- Episode completes without hanging

---

## 10. Calibration Refinement (Post-Bring-Up)

After initial runs, refine sensor intrinsics for better 3D projection:

1. **Capture calibration frames:** Run a short session collecting depth+RGB pairs
2. **Run OpenCV calibration:** Use standard pinhole model with checkerboard
3. **Update intrinsics:** Patch new fx/fy/cx/cy into config
4. **Validate:** Rerun smoke test and verify 3D points align with workspace

---

## 11. Hardware-In-The-Loop (HIL) Diagnostics

Use these tests to validate frame math, extrinsics behavior, dispatch safety observability,
step drift, and a practical stay-on-object proxy before long unsupervised runs.

### HIL Test Gating

All HIL tests are opt-in and skipped by default.

```bash
export TBP_ENABLE_HIL=1
```

Optional controls:

```bash
export TBP_HIL_ROBOT_IP=192.168.1.159
export TBP_HIL_USB_PORT=/dev/sipeed
export TBP_HIL_SETTLE_TIME_S=0.25
export TBP_HIL_STEP_DISTANCE_M=0.01
export TBP_HIL_SENSOR_TIMEOUT_S=0.15
export TBP_HIL_SENSOR_RETRIES=6
export TBP_HIL_SENSOR_RETRY_DELAY_S=0.10

# Required for tests that move hardware:
export TBP_HIL_ALLOW_MOTION=1

# Required for stay-on-object proxy checks:
export TBP_HIL_OBJECT_PRESENT=1
```

### Recommended Sequence

1. Preflight checks:

```bash
bash tools/real_world_preflight.sh
```

2. Run mocked regression first:

```bash
python -m pytest tests/integration/frameworks/environments/real_world_lite6_a010_smoke_test.py -v
```

3. Run HIL diagnostics (safe/default tier):

```bash
python -m pytest tests/integration/frameworks/environments/real_world_lite6_a010_hil_test.py -m hil -v -s -n 0
```

4. Run motion tier (strictly supervised):

```bash
TBP_HIL_ALLOW_MOTION=1 python -m pytest tests/integration/frameworks/environments/real_world_lite6_a010_hil_test.py -m "hil and requires_motion" -v -s -n 0
```

> HIL tests must run single-process (`-n 0`) to avoid parallel workers competing for `/dev/sipeed`.

### What These HIL Tests Verify

- **Frame rigidness:** `world_camera` rotation is orthonormal with determinant ~ +1.
- **Extrinsics usage:** Sensor translation offset appears consistently in world-camera and proprioceptive sensor pose.
- **Dispatch observability:** Out-of-workspace commands produce explicit rejection details (`reason_code`, `details`).
- **Step drift bound:** Observed end-effector step remains within a config-derived envelope from `max_translation_step_m`.
- **Stay-on-object proxy:** Valid-depth ratio recovers within a bounded number of steps when temporarily degraded.

### Initial Threshold Guidance

Thresholds are intentionally config-derived first:

- Drift envelope starts from `goal_adapter_config.max_translation_step_m`.
- Recovery defaults are controlled by:
  - `TBP_HIL_LOW_VALID_RATIO` (default `0.02`)
  - `TBP_HIL_RECOVER_VALID_RATIO` (default `0.05`)
  - `TBP_HIL_RECOVERY_STEPS` (default `3`)

After collecting baseline runs for your setup, tighten these values for stronger guarantees.

---

## References & Further Support

- **Maixsense A010 API:** `multimodal_monty_meets_world/maixsense_a010_api/README.md`
- **UFactory Lite6 Interface:** `multimodal_monty_meets_world/ufactory_api/monty_goal_adapter.py` (safety config docstring)
- **Real-World Environment:** `src/tbp/monty/frameworks/environments/real_world_environment.py` (hard-stop reason codes)
- **Hydra Configuration:** `src/tbp/monty/conf/environment/real_world_lite6_maixsense.yaml`

**Support Contact:** Check existing RFCs and documentation in `rfcs/` for design rationale and future work.

---

**End of Runbook**
