# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monty is a sensorimotor learning system from the Thousand Brains Project, inspired by neocortical principles. It models objects through active sensing — moving sensors over objects to build and recognize 3D models using features at poses, not passive image classification.

IMPORTANT: You will specifically be working on the implementation of Monty in a real world sensorimotor learning experiment. In this experiment, Monty is able to interface with a Ufactory Lite 6 robot, a Maixsense A010 ToF sensor (100x100 resolution with 8-bit precision with 70°(H) * 60°(V) FOV) mounted to the robots end effector, and a Zed 2i stereo camera. 
The robot and ToF sensor make up what is known as the "surface agent" and is intended to follow the surface of an object at a set distance, aligned vertically with the surface normal.
The stereo camera is part of the "distant agent" which is intended to capture an RGBD image of the object from one angle and form a model of that side of the object.
The real world objects that are being modelled are about the size of a mug/Rubik's cube.

## Build & Environment Setup

The project uses **conda** for environment management (see `environment.yml`). There is an experimental `uv` setup (`UV_PROTOTYPE.md`) but it is **not supported** — do not use `uv` unless explicitly asked.

```bash
# Install in development mode
pip install -e ".[dev]"
```

## Common Commands

```bash
# Run all unit + conf tests (default pytest config uses -n auto for parallel)
pytest

# Run a single test file
pytest tests/unit/frameworks/test_file.py

# Run a single test
pytest tests/unit/frameworks/test_file.py::TestClass::test_method

# Lint check
ruff check

# Auto-fix lint issues
ruff check --fix

# Format code
ruff format

# Type checking
mypy src/

# Run an experiment (Hydra-based)
python run.py experiment=<experiment_name>
```

## Test Structure

- `tests/unit/` — Unit tests (main test suite)
- `tests/conf/` — Configuration snapshot tests (validates YAML configs haven't changed unexpectedly; run `python src/tbp/monty/conf/update_snapshots.py` to update snapshots after intentional config changes)
- `tests/integration/` — Integration tests (not in default pytest paths; must be run explicitly)
- Default pytest runs `tests/conf` and `tests/unit` with `pytest-xdist` parallelism (`-n auto`)
- Markers: `hil` for hardware-in-the-loop tests, `requires_motion` for robot movement tests

## Architecture

### Core Abstractions (`src/tbp/monty/frameworks/models/abstract_monty_classes.py`)

The system is built from composable abstract classes:
- **Monty** — Top-level orchestrator connecting sensor modules, learning modules, and motor systems
- **SensorModule (SM)** — Transforms raw sensor data into a common format (features at poses)
- **LearningModule (LM)** — Builds and matches object models from sensory input; the evidence-based LM is the primary implementation (`models/evidence_matching/`)
- **MotorSystem** — Translates abstract motor commands into agent-specific actions
- **GoalGenerator** — LMs generate goals (desired movements) for the motor system

### Cortical Messaging Protocol (CMP) (`src/tbp/monty/cmp.py`)

All inter-component communication uses CMP messages — a unified format of **features at poses**. Messages flow between SMs, LMs, and motor systems. This enables arbitrary topologies of components (hierarchy, heterarchy, voting).

### Experiment Framework

- Experiments are configured via **Hydra** YAML configs in `src/tbp/monty/conf/`
- Entry point: `python run.py experiment=<name>` which loads `src/tbp/monty/conf/experiment.yaml` as the base config
- Experiment classes in `src/tbp/monty/frameworks/experiments/` control train/eval workflow
- Config hierarchy: `conf/experiment/` for experiment definitions, `conf/monty/` for Monty configs, `conf/environment/` for environment configs

### Environments & Simulators

- `src/tbp/monty/simulators/` — Simulator backends (Habitat for 3D scenes, MuJoCo for physics)
- `src/tbp/monty/frameworks/environments/` — Environment interfaces that wrap simulators
- Real-world hardware support via `real_world_environment.py` and `real_world_interface.py`

### Key Directories

- `src/tbp/monty/frameworks/models/motor_policies.py` — Movement policies (exploration, goal-directed)
- `src/tbp/monty/frameworks/models/sensor_modules.py` — SM implementations
- `src/tbp/monty/frameworks/models/object_model.py` — Object model representation
- `multimodal_monty_meets_world/maixsense_a010_api/monty_adapter.py` — ToF Sensor adapter for Monty
- `multimodal_monty_meets_world/ufactory_api/monty_goal_adapter.py` — Robot goal adapter for Monty
- `multimodal_monty_meets_world/ufactory_api/robot_interface.py` — Robot interface
- `multimodal_monty_meets_world/` — More files relating to the hardware in use
- `src/tbp/monty/conf/experiment/real_world/` — Main experiment folder for the real world implementation
- `benchmarks/` — Benchmark experiment results and analysis

## Code Conventions

- **NumPy over PyTorch**: Use NumPy for all vector/matrix operations. PyTorch is only used for multiprocessing and specific utilities. This is a deliberate project policy — avoid introducing PyTorch-based solutions.
- **Line length**: 88 characters
- **Docstrings**: Google style
- **Linter/Formatter**: Ruff (configured in `pyproject.toml`)
- **Copyright header**: All source files must have the MIT license header with `Thousand Brains Project` copyright. When modifying existing files, add the current year to the copyright if not already present.
- **Abstract classes**: Should contain no implementation — only interface definitions. The project intends to migrate to Protocols.

# Token efficiency
Respond like smart caveman. Cut all filler, keep technical substance.
- Drop articles (a, an, the), filler (just, really, basically, actually).
- Drop pleasantries (sure, certainly, happy to).
- No hedging. Fragments fine. Short synonyms.
- Technical terms stay exact. Code blocks unchanged.
- Pattern: [thing] [action] [reason]. [next step].