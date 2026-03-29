#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROBOT_IP="${ROBOT_IP:-192.168.1.159}"
USB_PORT="${USB_PORT:-/dev/sipeed}"
ENV_CFG="${ENV_CFG:-src/tbp/monty/conf/environment/real_world_lite6_maixsense.yaml}"
EXPERIMENT_CFG="${EXPERIMENT_CFG:-src/tbp/monty/conf/experiment/real_world/lite6_maixsense_unsupervised.yaml}"
IFACE_CFG="${IFACE_CFG:-src/tbp/monty/conf/env_interface/real_world_manual_train_eval.yaml}"

log() {
  printf '[preflight] %s\n' "$1"
}

fail() {
  printf '[preflight][FAIL] %s\n' "$1" >&2
  exit 1
}

pass() {
  printf '[preflight][OK] %s\n' "$1"
}

log "Starting real-world preflight checks (no robot motion)"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pass "Activated virtual environment (.venv)"
else
  log "No .venv found; using current Python environment"
fi

command -v python >/dev/null 2>&1 || fail "python is not available in PATH"
pass "Python executable found"

[[ -f "$ENV_CFG" ]] || fail "Missing environment config: $ENV_CFG"
[[ -f "$EXPERIMENT_CFG" ]] || fail "Missing experiment config: $EXPERIMENT_CFG"
[[ -f "$IFACE_CFG" ]] || fail "Missing env interface config: $IFACE_CFG"
pass "Required config files exist"

if ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; then
  pass "Robot IP reachable ($ROBOT_IP)"
else
  fail "Robot IP not reachable ($ROBOT_IP). Check network/power before running experiments"
fi

if [[ -e "$USB_PORT" ]]; then
  pass "USB device path exists ($USB_PORT)"
else
  fail "USB device path not found ($USB_PORT). Check cable/udev symlink"
fi

if [[ -r "$USB_PORT" && -w "$USB_PORT" ]]; then
  pass "USB device has read/write access ($USB_PORT)"
else
  fail "USB device lacks read/write access ($USB_PORT). Check udev rules/group permissions"
fi

python - <<'PY'
import importlib
from pathlib import Path

from omegaconf import OmegaConf

required_modules = [
    "hydra",
    "omegaconf",
    "numpy",
    "quaternion",
    "serial",
    "xarm.wrapper",
    "multimodal_monty_meets_world.factory",
    "tbp.monty.frameworks.environments.real_world_environment",
]

for mod in required_modules:
    importlib.import_module(mod)

env_cfg = OmegaConf.load("src/tbp/monty/conf/environment/real_world_lite6_maixsense.yaml")
iface_cfg = OmegaConf.load("src/tbp/monty/conf/env_interface/real_world_manual_train_eval.yaml")
exp_cfg = OmegaConf.load("src/tbp/monty/conf/experiment/real_world/lite6_maixsense_unsupervised.yaml")

env_init_args = env_cfg.get("env_init_args")
if env_init_args is None:
    raise RuntimeError("env_init_args is missing from environment config")

required_env_targets = [
    ("robot_interface", "multimodal_monty_meets_world.factory.create_robot_interface"),
    ("sensor_client", "multimodal_monty_meets_world.factory.create_usb_frame_client"),
    ("observation_adapter", "multimodal_monty_meets_world.factory.create_observation_adapter"),
]

for key, target in required_env_targets:
    node = env_init_args.get(key)
    if node is None:
        raise RuntimeError(f"{key} is missing in env_init_args")
    actual = node.get("_target_")
    if actual != target:
        raise RuntimeError(f"{key} target mismatch: expected {target}, got {actual}")

home_pose = env_init_args.get("home_pose_mm_deg")
if home_pose is None or len(home_pose) != 6:
    raise RuntimeError("home_pose_mm_deg must exist and have 6 values")

for cfg_key in ["fx", "fy", "cx", "cy"]:
    if env_init_args.observation_adapter.get(cfg_key) is None:
        raise RuntimeError(f"observation_adapter missing {cfg_key}")

if iface_cfg.get("do_train") is not True or iface_cfg.get("do_eval") is not True:
    raise RuntimeError("env interface must enable both do_train and do_eval")

for side in ["train_env_interface_args", "eval_env_interface_args"]:
    if iface_cfg.get(side, {}).get("use_goal_pose_dispatch") is not True:
        raise RuntimeError(f"{side}.use_goal_pose_dispatch must be true")

defaults = exp_cfg.get("defaults", [])
default_text = "\n".join(str(d) for d in defaults)
if "/environment: real_world_lite6_maixsense" not in default_text:
    raise RuntimeError("experiment defaults do not select real_world_lite6_maixsense environment")
if "/env_interface: real_world_manual_train_eval" not in default_text:
    raise RuntimeError("experiment defaults do not select real_world_manual_train_eval interface")

print("[preflight][OK] Python imports and config semantics validated")
PY

pass "All preflight checks passed. Safe to start first controlled hardware run"
log "Next: run your short experiment with low step counts for bring-up"
