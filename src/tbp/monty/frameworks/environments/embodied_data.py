# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
import logging
from pprint import pformat
from typing import Callable, Mapping, Sequence, cast

import numpy as np
import quaternion as qt
from omegaconf import ListConfig

from tbp.monty.frameworks.actions.actions import (
    Action,
)
from tbp.monty.frameworks.environment_utils.transforms import (
    Transform,
    TransformContext,
)
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    SemanticID,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.environments.object_init_samplers import (
    Default,
    MultiObjectNames,
    ObjectInitParams,
    Predefined,
    RandomRotation,
)
from tbp.monty.frameworks.environments.positioning_procedures import (
    PositioningProcedureFactory,
)
from tbp.monty.frameworks.environments.two_d_data import (
    OmniglotEnvironment,
    SaccadeOnImageEnvironment,
    SaccadeOnImageFromStreamEnvironment,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import (
    MotorSystemState,
    ProprioceptiveState,
)
from tbp.monty.frameworks.utils.incoming_scenes_manager import IncomingScenesManager
from tbp.monty.frameworks.utils.zed_camera_capture import ZEDRGBDCapture

__all__ = [
    "EnvironmentInterface",
    "EnvironmentInterfacePerObject",
    "OmniglotEnvironmentInterface",
    "SaccadeOnImageEnvironmentInterface",
    "SaccadeOnImageFromStreamEnvironmentInterface",
]

logger = logging.getLogger(__name__)


def normalize_transforms(
    transform: Transform | Sequence[Transform] | None,
) -> tuple[Transform, ...]:
    """Normalize transform configuration to a tuple of transforms.

    Returns:
        A tuple of transforms.
    """
    if transform is None:
        return ()
    if callable(transform):
        return (transform,)
    return tuple(transform)


class EnvironmentInterface:
    """Provides an interface to an embodied environment.

    Observations and proprioceptive state are returned from the environment
    based on the actions taken.

    Attributes:
        env: An instance of a class that implements :class:`SimulatedObjectEnvironment`.
        rng: Random number generator to use.
        seed: The configured random seed.
        experiment_mode: The experiment mode that this environment interface is used
            in.
        transform: Callable used to transform the observations returned by
            the environment.

    Note:
        This one on its own won't work.
    """

    def __init__(
        self,
        env: SimulatedObjectEnvironment,
        rng: np.random.RandomState,
        seed: int,
        experiment_mode: ExperimentMode,
        transform: Transform | Sequence[Transform] | None = None,
    ):
        self.env = env
        self.rng = rng
        self.seed = seed
        self.transforms = normalize_transforms(transform)
        self.reset(self.rng)
        self.experiment_mode = experiment_mode

    def reset(self, rng: np.random.RandomState):
        self.rng = rng
        observations, state = self.env.reset()

        if self.transforms:
            observations = self.apply_transform(observations, state)
        return observations, state

    def apply_transform(
        self,
        observations: Observations,
        state: ProprioceptiveState,
    ) -> Observations:
        ctx = TransformContext(rng=self.rng, state=state)
        for transform in self.transforms:
            observations = transform(observations, ctx)
        return observations

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Take actions in the environment and apply the transform to the observations.

        Args:
            actions: The actions to take in the environment.

        Returns:
            The observations and proprioceptive state.
        """
        observations, state = self.env.step(actions)
        if self.transforms is not None:
            observations = self.apply_transform(observations, state)
        return observations, state

    def pre_episode(self, rng: np.random.RandomState):
        self.reset(rng)

    def post_episode(self):
        pass

    def pre_epoch(self):
        pass

    def post_epoch(self):
        pass


class EnvironmentInterfacePerObject(EnvironmentInterface):
    """Interface for testing in an environment with one "primary target" object.

    Interface for testing in an environment where we load one "primary target" object
    at a time; in addition, we can optionally add other "distractor" objects to the
    environment.

    Has a list of primary target objects, swapping these objects in and out for episodes
    without resetting the environment. The objects are initialized with parameters such
    that we can vary their location, rotation, and scale.

    After the primary target is added to the environment, other distractor objects,
    sampled from the same object list, can be added.
    """

    def __init__(
        self,
        object_names: list[str] | ListConfig | MultiObjectNames,
        object_init_sampler: Default | Predefined | RandomRotation,
        parent_to_child_mapping: Mapping[str, Sequence[str]] | None = None,
        positioning_procedures: Sequence[PositioningProcedureFactory] | None = None,
        *args,
        **kwargs,
    ):
        """Initialize environment interface.

        Args:
            object_names: plain list of objects, or Hydra `ListConfig`, if doing a
                simple experiment with primary target objects only; mapping typed as
                `MultiObjectNames` for experiments with multiple objects,
                corresponding to -->
                targets_list : the list of primary target objects
                source_object_list : the original object list from which the primary
                    target objects were sampled; used to sample distractor objects
                num_distractors : the number of distractor objects to add to the
                    environment
            object_init_sampler: Function that returns dict with position, rotation,
                and scale of objects when re-initializing.
            parent_to_child_mapping: dictionary mapping parent objects to their child
                objects. Used for logging.
            positioning_procedures: Sequence of positioning procedures to apply
                prior to each episode.
            *args: passed to `super()` call
            **kwargs: passed to `super()` call

        Raises:
            TypeError: If `object_names` is not a `list`, `ListConfig`, or a mapping
        """
        super().__init__(*args, **kwargs)
        if isinstance(object_names, Mapping):
            # TODO when we want more advanced multi-object experiments, update these
            # arguments along with the Object Initializers so that we can easily
            # specify a set of primary targets and distractors, i.e. random sampling
            # of the distractor objects shouldn't happen here
            self.object_names = object_names["targets_list"]
            self.source_object_list = list(
                dict.fromkeys(object_names["source_object_list"])
            )
            self.num_distractors = object_names["num_distractors"]
        elif isinstance(object_names, (list, ListConfig)):
            self.object_names = object_names
            # Return an (ordered) list of unique items:
            self.source_object_list = list(dict.fromkeys(self.object_names))
            self.num_distractors = 0
        else:
            raise TypeError("Object names must be a list, ListConfig, or a mapping")
        self.create_semantic_mapping()

        self.episodes = 0
        self.epochs = 0
        self.object_init_sampler = object_init_sampler
        self.object_params: ObjectInitParams = self.object_init_sampler(
            self.seed, self.experiment_mode, self.epochs, self.episodes
        )
        self.current_object = 0
        self.n_objects = len(self.object_names)
        self.primary_target = None
        self.consistent_child_objects = None
        self.parent_to_child_mapping = parent_to_child_mapping or {}
        self._positioning_procedures = positioning_procedures

    def pre_episode(self, rng: np.random.RandomState):
        super().pre_episode(rng)

        if self._positioning_procedures is None:
            return

        assert self.primary_target is not None, "Primary target not set"
        target_semantic_id = cast("SemanticID", self.primary_target["semantic_id"])

        success = False
        for factory in self._positioning_procedures:
            positioning_procedure = factory.create(target_semantic_id)
            observations, proprioceptive_state = self.step([])
            result = positioning_procedure(
                observations, MotorSystemState(proprioceptive_state)
            )
            while not result.terminated and not result.truncated:
                observations, proprioceptive_state = self.step(result.actions)
                result = positioning_procedure(
                    observations, MotorSystemState(proprioceptive_state)
                )

            # We only care about the last result.
            success = result.success

        if self.num_distractors == 0 and not success:
            raise RuntimeError("Primary target not visible at start of episode")

    def post_episode(self):
        super().post_episode()
        self.episodes += 1
        self.object_params = self.object_init_sampler(
            self.seed, self.experiment_mode, self.epochs, self.episodes
        )
        self.cycle_object()

    def pre_epoch(self):
        self.change_object_by_idx(0)

    def post_epoch(self):
        self.epochs += 1
        self.object_params = self.object_init_sampler(
            self.seed, self.experiment_mode, self.epochs, self.episodes
        )

    def create_semantic_mapping(self):
        """Create a unique semantic ID (positive integer) for each object.

        Used by Habitat for the semantic sensor.

        In addition, create a dictionary mapping back and forth between these IDs and
        the corresponding name of the object
        """
        assert set(self.object_names).issubset(set(self.source_object_list)), (
            "Semantic mapping requires primary targets sampled from source list"
        )

        starting_integer = 1  # Start at 1 so that we can distinguish on-object semantic
        # IDs (>0) from being off object (semantic_id == 0 in Habitat by default)
        self.semantic_id_to_label = {
            SemanticID(i + starting_integer): label
            for i, label in enumerate(self.source_object_list)
        }
        self.semantic_label_to_id = {
            label: SemanticID(i + starting_integer)
            for i, label in enumerate(self.source_object_list)
        }

    def cycle_object(self):
        """Remove the previous object(s) from the scene and add a new primary target.

        Also add any potential distractor objects.
        """
        next_object = (self.current_object + 1) % self.n_objects
        logger.info(
            f"\n\nGoing from {self.current_object} to {next_object} of {self.n_objects}"
        )
        self.change_object_by_idx(next_object)

    def change_object_by_idx(self, idx: int):
        """Update the primary target object in the scene based on the given index.

        The given `idx` is the index of the object in the `self.object_names` list,
        which should correspond to the index of the object in the `self.object_params`
        list.

        Also add any distractor objects if required.

        Args:
            idx: Index of the new object and its parameters in object_params

        Raises:
            IndexError: If idx is outside the range [0, self.n_objects).
        """
        if not 0 <= idx < self.n_objects:
            raise IndexError(f"idx must satisfy 0 <= idx < {self.n_objects}, got {idx}")
        self.env.remove_all_objects()

        # Specify config for the primary target object and then add it
        init_params = self.object_params.copy()
        init_params.pop("euler_rotation")
        if "quat_rotation" in init_params:
            init_params.pop("quat_rotation")
        init_params["semantic_id"] = self.semantic_label_to_id[self.object_names[idx]]

        # TODO clean this up with its own specific call i.e. Law of Demeter
        primary_target_obj = self.env.add_object(
            name=self.object_names[idx], **init_params
        )

        if self.num_distractors > 0:
            self.add_distractor_objects(
                primary_target_obj,
                init_params,
                primary_target_name=self.object_names[idx],
            )

        self.current_object = idx
        self.primary_target = {
            "object": self.object_names[idx],
            "semantic_id": self.semantic_label_to_id[self.object_names[idx]],
            **self.object_params,
        }
        if self.primary_target["object"] in self.parent_to_child_mapping:
            self.consistent_child_objects = self.parent_to_child_mapping[
                self.primary_target["object"]
            ]
        elif self.parent_to_child_mapping:
            # if mapping contains keys (i.e. not an empty dict) it should contain the
            # target object
            logger.warning(
                f"target object {self.primary_target['object']} not in",
                " parent_to_child_mapping",
            )
        logger.info(f"New primary target: {pformat(self.primary_target)}")

    def add_distractor_objects(
        self,
        primary_target_obj: ObjectID,
        init_params: ObjectInitParams,
        primary_target_name: str,
    ):
        """Add arbitrarily many "distractor" objects to the environment.

        Args:
            primary_target_obj : The ID of the object which is the primary target in
                the scene.
            init_params: Parameters used to initialize the object, e.g.
                orientation; for now, these are identical to the primary target
                except for the object ID.
            primary_target_name: name of the primary target object
        """
        # Sample distractor objects from those that are not the primary target; this
        # is so that, for now, we can evaluate how well the model stays on the primary
        # target object until it is classified, with no ambiguity about what final
        # object it is classifying
        sampling_list = [
            item for item in self.source_object_list if item != primary_target_name
        ]

        for __ in range(self.num_distractors):
            new_init_params = copy.deepcopy(init_params)

            new_obj_label = self.rng.choice(sampling_list)
            new_init_params["semantic_id"] = self.semantic_label_to_id[new_obj_label]
            # TODO clean up the `**` unpacking used
            self.env.add_object(
                name=new_obj_label,
                **new_init_params,
                primary_target_object=primary_target_obj,
            )


class OmniglotEnvironmentInterface(EnvironmentInterfacePerObject):
    """Environment interface for Omniglot dataset."""

    def __init__(
        self,
        alphabets: Sequence[int],
        characters: Sequence[int],
        versions: Sequence[int],
        env: OmniglotEnvironment,
        rng: np.random.RandomState,
        transform: Transform | Sequence[Transform] | None = None,
        parent_to_child_mapping: Mapping[str, Sequence[str]] | None = None,
        positioning_procedures: Sequence[PositioningProcedureFactory] | None = None,
        *_args,
        **_kwargs,
    ):
        """Initialize environment interface.

        Args:
            alphabets: List of alphabets.
            characters: List of characters.
            versions: List of versions.
            env: An instance of a class that implements :class:`OmniglotEnvironment`.
            rng: Random number generator to use.
            transform: Callable used to transform the observations returned
                 by the environment.
            parent_to_child_mapping: dictionary mapping parent objects to their child
                objects. Used for logging.
            positioning_procedures: Sequence of positioning procedures to apply
                prior to each episode.
            *args: Unused?
            **kwargs: Unused?
        """
        self.env = env
        self.rng = rng
        self.transforms = normalize_transforms(transform)
        self.reset(self.rng)

        self.alphabets = alphabets
        self.characters = characters
        self.versions = versions
        self.current_object = 0
        self.n_objects = len(characters)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None
        self.object_names = [
            str(self.env.alphabet_names[alphabets[i]]) + "_" + str(self.characters[i])
            for i in range(self.n_objects)
        ]
        self.consistent_child_objects = None
        self.parent_to_child_mapping = (
            parent_to_child_mapping if parent_to_child_mapping else {}
        )
        self._positioning_procedures = positioning_procedures

    def post_episode(self):
        self.cycle_object()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_object(self):
        """Switch to the next character image."""
        next_object = (self.current_object + 1) % self.n_objects
        logger.info(
            f"\n\nGoing from {self.current_object} to {next_object} of {self.n_objects}"
        )
        self.change_object_by_idx(next_object)

    def change_object_by_idx(self, idx: int):
        """Update the object in the scene given the idx of it in the object params.

        Args:
            idx: Index of the new object and ints parameters in object params

        Raises:
            IndexError: If idx is outside the range [0, self.n_objects).
        """
        if not 0 <= idx < self.n_objects:
            raise IndexError(f"idx must satisfy 0 <= idx < {self.n_objects}, got {idx}")
        self.env.switch_to_object(
            self.alphabets[idx], self.characters[idx], self.versions[idx]
        )
        self.current_object = idx
        self.primary_target = {
            "object": self.object_names[idx],
            "rotation": qt.quaternion(0, 0, 0, 1),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }


class SaccadeOnImageEnvironmentInterface(EnvironmentInterfacePerObject):
    """Environment interface for moving over a 2D image with depth channel."""

    def __init__(
        self,
        scenes: Sequence[int],
        versions: Sequence[int],
        env: SaccadeOnImageEnvironment,
        rng: np.random.RandomState,
        transform: Transform | Sequence[Transform] | None = None,
        parent_to_child_mapping: Mapping[str, Sequence[str]] | None = None,
        enable_manual_scene_picker: bool = False,
        scene_picker_prompt: str = "Select next scene/version (e.g. '0 1'), 'keep', or 'quit': ",
        episodes_per_epoch: int | None = None,
        allow_dynamic_scene_refresh: bool = False,
        enable_scene_picker_captures: bool = False,
        scene_capture_prefix: str = "zed_capture",
        input_fn: Callable[[str], str] | None = None,
        positioning_procedures: Sequence[PositioningProcedureFactory] | None = None,
        *_args,
        **_kwargs,
    ):
        """Initialize environment interface.

        Args:
            scenes: List of scenes
            versions: List of versions
            env: An instance of a class that implements
                :class:`SaccadeOnImageEnvironment`.
            rng: Random number generator to use.
            transform: Callable used to transform the observations returned by
                the environment.
            parent_to_child_mapping: dictionary mapping parent objects to their child
                objects. Used for logging.
            enable_manual_scene_picker: Whether to prompt for scene/version selection
                after each episode.
            scene_picker_prompt: Prompt shown for manual scene/version selection.
            episodes_per_epoch: Fixed number of episodes to run per epoch.
            allow_dynamic_scene_refresh: Whether to refresh available scene/version
                options from disk during a run.
            enable_scene_picker_captures: Whether to allow triggering ZED captures
                from the scene picker prompt.
            scene_capture_prefix: Prefix for new auto-indexed capture scene folders.
            input_fn: Input function used for terminal prompts. Defaults to `input`.
            positioning_procedures: Sequence of positioning procedures to apply
                prior to each episode.
            *args: Unused?
            **kwargs: Unused?
        """
        self.env = env
        self.rng = rng
        self.transforms = normalize_transforms(transform)
        self.reset(self.rng)

        self.scenes = list(scenes)
        self.versions = list(versions)
        self.object_names = self.env.scene_names
        if len(self.scenes) != len(self.versions):
            raise ValueError("`scenes` and `versions` must have the same length")
        self.n_versions = len(self.versions)
        self.current_scene_version = self.n_versions - 1
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None
        self.consistent_child_objects = None
        self.parent_to_child_mapping = parent_to_child_mapping or {}
        self.enable_manual_scene_picker = enable_manual_scene_picker
        self.scene_picker_prompt = scene_picker_prompt
        self.allow_dynamic_scene_refresh = allow_dynamic_scene_refresh
        self.enable_scene_picker_captures = enable_scene_picker_captures
        self.scene_capture_prefix = scene_capture_prefix
        self.input_fn = input_fn or input
        self._positioning_procedures = positioning_procedures

        if self.n_versions == 0 or self.allow_dynamic_scene_refresh:
            self.refresh_scene_schedule_from_env()
        if self.n_versions == 0:
            raise RuntimeError("No scene/version pairs available for Saccade interface")

        self.episodes_per_epoch = episodes_per_epoch or self.n_versions
        if self.episodes_per_epoch <= 0:
            raise ValueError("'episodes_per_epoch' must be a positive integer")

        # Set an initial target for the very first episode.
        self.change_object_by_idx(self.current_scene_version)

    def _capture_scene_from_picker(self) -> bool:
        """Capture RGBD data and persist it as a new scene or new scene version."""
        capture_confirmation = self.input_fn(
            "Capture from ZED now? [y/N]: "
        ).strip().lower()
        if capture_confirmation not in {"y", "yes"}:
            logger.info("Capture cancelled by user")
            return False

        target_choice = self.input_fn(
            "Capture as new [s]cene, new [v]ersion, or [c]ancel: "
        ).strip().lower()
        if target_choice in {"c", "cancel", ""}:
            logger.info("Capture cancelled by user")
            return False

        manager = IncomingScenesManager(self.env.data_path)

        try:
            with ZEDRGBDCapture() as capture:
                if not capture.is_available():
                    logger.warning("ZED camera unavailable; skipping capture")
                    return False

                rgb_image, depth_array, metadata = capture.grab_single_frame()
                if rgb_image is None or depth_array is None:
                    logger.warning("Failed to capture RGBD frame from ZED")
                    return False

                if target_choice in {"s", "scene"}:
                    scene_name = manager.create_next_scene_name(
                        prefix=self.scene_capture_prefix
                    )
                    scene_path = manager.create_scene_folder(scene_name)
                    version = 0
                elif target_choice in {"v", "version"}:
                    self.refresh_scene_schedule_from_env()
                    options = [
                        f"{idx}: {scene_name}"
                        for idx, scene_name in enumerate(self.env.scene_names)
                    ]
                    options_text = "\n".join(options)
                    selection = self.input_fn(
                        "Select scene index for new version:\n"
                        f"{options_text}\n"
                        "Scene index: "
                    ).strip()
                    try:
                        scene_idx = int(selection)
                    except ValueError:
                        logger.warning(
                            "Invalid scene index '%s'; capture cancelled", selection
                        )
                        return False

                    if scene_idx < 0 or scene_idx >= len(self.env.scene_names):
                        logger.warning(
                            "Scene index %s out of range [0, %s]",
                            scene_idx,
                            len(self.env.scene_names) - 1,
                        )
                        return False

                    scene_name = self.env.scene_names[scene_idx]
                    scene_path = manager.resolve_scene_path(scene_name)
                    version = manager.get_next_version_index(scene_path)
                else:
                    logger.warning(
                        "Unknown capture target '%s'; expected n/v/c", target_choice
                    )
                    return False

                saved_paths = manager.save_rgbd_capture(
                    scene_path=scene_path,
                    version=version,
                    rgb_image=rgb_image,
                    depth_array=depth_array,
                    metadata=metadata,
                )

            self.refresh_scene_schedule_from_env()
            logger.info(
                "Captured scene='%s', version=%s (rgb=%s, depth=%s)",
                scene_name,
                version,
                saved_paths["rgb_path"],
                saved_paths["depth_path"],
            )
            return True
        except Exception as exc:
            logger.warning("Scene picker capture failed: %s", exc)
            return False

    def refresh_scene_schedule_from_env(self) -> None:
        """Refresh scene/version schedule from files discovered by the environment."""
        if hasattr(self.env, "refresh_scene_catalog"):
            self.env.refresh_scene_catalog()

        if not hasattr(self.env, "get_scene_version_pairs"):
            return

        pairs = self.env.get_scene_version_pairs()
        if len(pairs) == 0:
            raise RuntimeError("No valid scene/version pairs found in environment data")

        self.object_names = self.env.scene_names
        self.scenes = [scene_idx for scene_idx, _, _ in pairs]
        self.versions = [version_id for _, version_id, _ in pairs]
        self.n_versions = len(self.versions)
        if self.current_scene_version >= self.n_versions:
            self.current_scene_version = self.n_versions - 1

    def _prompt_for_next_scene_selection(self) -> None:
        """Prompt operator to choose scene/version for the next episode."""
        while True:
            if self.allow_dynamic_scene_refresh:
                self.refresh_scene_schedule_from_env()

            options = [
                f"{idx}: scene={self.scenes[idx]} ({self.object_names[self.scenes[idx]]}), "
                f"version={self.versions[idx]}"
                for idx in range(self.n_versions)
            ]
            options_text = "\n".join(options)
            prompt = (
                f"\n{self.scene_picker_prompt}\n"
                "Options:\n"
                f"{options_text}\n"
                f"Current option index: {self.current_scene_version}\n"
                + (
                    "Extra commands: capture/new/c\n"
                    if self.enable_scene_picker_captures
                    else ""
                )
                +
                "Selection: "
            )
            selection = self.input_fn(prompt).strip().lower()

            if selection in {"", "keep", "k"}:
                logger.info(
                    "Keeping current scene selection index %s", self.current_scene_version
                )
                return
            if selection in {"quit", "q", "exit"}:
                raise KeyboardInterrupt("Stopped by user during scene selection")

            if self.enable_scene_picker_captures and selection in {
                "capture",
                "new",
                "c",
            }:
                self._capture_scene_from_picker()
                continue

            try:
                idx = int(selection)
            except ValueError:
                logger.warning("Invalid selection '%s'. Enter an option index.", selection)
                continue

            if idx < 0 or idx >= self.n_versions:
                logger.warning(
                    "Selection %s is out of range [0, %s]", idx, self.n_versions - 1
                )
                continue

            self.change_object_by_idx(idx)
            return

    def pre_epoch(self) -> None:
        if self.enable_manual_scene_picker and self.epochs == 0:
            self._prompt_for_next_scene_selection()

    def post_episode(self):
        if self.enable_manual_scene_picker:
            self._prompt_for_next_scene_selection()
        else:
            self.cycle_object()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_object(self):
        """Switch to the next scene image."""
        if self.allow_dynamic_scene_refresh:
            self.refresh_scene_schedule_from_env()

        next_scene = (self.current_scene_version + 1) % self.n_versions
        logger.info(
            f"\n\nGoing from {self.current_scene_version} to {next_scene} of "
            f"{self.n_versions}"
        )
        self.change_object_by_idx(next_scene)

    def change_object_by_idx(self, idx: int):
        """Update the object in the scene given the idx of it in the object params.

        Args:
            idx: Index of the new object and ints parameters in object params

        Raises:
            IndexError: If idx is outside the range [0, self.n_versions).
        """
        if not 0 <= idx < self.n_versions:
            raise IndexError(
                f"idx must satisfy 0 <= idx < {self.n_versions}, got {idx}"
            )
        logger.info(
            f"changing to obj {idx} -> scene {self.scenes[idx]}, version "
            f"{self.versions[idx]}"
        )
        self.env.switch_to_object(self.scenes[idx], self.versions[idx])
        self.current_scene_version = idx
        # TODO: Currently not differentiating between different poses/views
        target_object = self.object_names[self.scenes[idx]]
        # remove scene index from name
        target_object_formatted = "_".join(target_object.split("_")[1:])
        self.primary_target = {
            "object": target_object_formatted,
            "rotation": qt.quaternion(0, 0, 0, 1),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }


class SaccadeOnImageFromStreamEnvironmentInterface(SaccadeOnImageEnvironmentInterface):
    """Environment interface for moving over a 2D image with depth channel."""

    def __init__(
        self,
        env: SaccadeOnImageFromStreamEnvironment,
        rng: np.random.RandomState,
        transform: Transform | Sequence[Transform] | None = None,
        positioning_procedures: Sequence[PositioningProcedureFactory] | None = None,
        *_args,
        **_kwargs,
    ):
        """Initialize environment interface.

        Args:
            env: An instance of a class that implements
                :class:`SaccadeOnImageFromStreamEnvironment`.
            rng: Random number generator to use.
            transform: Callable used to transform the observations returned by
                the environment.
            positioning_procedures: Sequence of positioning procedures to apply
                prior to each episode.
            *args: Unused?
            **kwargs: Unused?
        """
        self.env = env
        self.rng = rng
        self.transforms = normalize_transforms(transform)
        self.reset(self.rng)
        self.current_scene = 0
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None
        self._positioning_procedures = positioning_procedures

    def pre_epoch(self):
        # TODO: Could give a start index as parameter
        self.change_scene_by_idx(0)

    def post_episode(self):
        self.cycle_scene()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_scene(self):
        """Switch to the next scene image."""
        next_scene = self.current_scene + 1
        logger.info(f"\n\nGoing from {self.current_scene} to {next_scene}")
        # TODO: Do we need a separate method for this ?
        self.change_scene_by_idx(next_scene)

    def change_scene_by_idx(self, idx: int):
        """Update the object in the scene given the idx of it in the object params.

        Args:
            idx: Index of the new object and ints parameters in object params
        """
        logger.info(f"changing to scene {idx}")
        self.env.switch_to_scene(idx)
        self.current_scene = idx
        # TODO: Currently not differentiating between different poses/views
        # TODO: Are the targets important here ? How can we provide the proper
        # targets corresponding to the current scene ?
        self.primary_target = {
            "object": "no_label",
            "rotation": qt.quaternion(0, 0, 0, 1),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }
