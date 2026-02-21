from typing import Tuple
import jax.numpy as jnp
from abc import ABC, abstractmethod
from flax import struct


@struct.dataclass
class ReferenceTrajectory: 
    ref_xs: jnp.ndarray  # (horizon+1, n_x)
    ref_us: jnp.ndarray  # (horizon, n_u)
    lambdas: jnp.ndarray  # (horizon, n_constraints)


@struct.dataclass
class ObstacleData:
    obs_pos: jnp.ndarray
    obs_r: jnp.ndarray

# Constraint 
@struct.dataclass
class ConstraintArgs:
    obs_data1: ObstacleData
    obs_data2: ObstacleData
    obs_data3: ObstacleData


class ControllerBase(ABC):
    def __init__(self, horizon: int = 10):
        self.horizon: int = horizon
        pass

    @abstractmethod
    def calc_input(self, x: jnp.ndarray, target_x: jnp.ndarray, reference_trajectory: ReferenceTrajectory) -> Tuple[jnp.ndarray, ReferenceTrajectory]:

        raise NotImplementedError()