from typing import Dict, List, Callable, Tuple, Any
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod


class DynamicsBase(ABC):
    def __init__(self, dt: float, nx: int, nu: int):
        self.nx: int = nx
        self.nu: int = nu
        self.dt: float = dt
        pass

    @abstractmethod
    def get_dynamics(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        raise NotImplementedError()

# ----------------------------
class TurtleModel(DynamicsBase):
    def __init__(self, dt: float):
        super(TurtleModel, self).__init__(dt, nx=5, nu=2)

    def get_dynamics(self):
        def turtle_dynamics(state: jnp.ndarray,
                            control: jnp.ndarray) -> jnp.ndarray:
            """
            state  = [x, y, theta, v_prev, omega_prev]
            control = [v, omega]
            """

            # Average current and previous control (actuator smoothing)
            u_eff = 0.5 * (control + state[3:])

            v = u_eff[0]
            omega = u_eff[1]
            theta = state[2]

            # Kinematic update
            x_new = state[0] + v * self.dt * jnp.cos(theta)
            y_new = state[1] + v * self.dt * jnp.sin(theta)
            theta_new = theta + omega * self.dt

            # Store current control in state
            new_state = jnp.array([
                x_new,
                y_new,
                theta_new,
                control[0],
                control[1]
            ])

            return new_state

        return jax.jit(turtle_dynamics)


class ParallelTwoWheelVehicleModel(DynamicsBase):
    def __init__(self, dt: float):
        super(ParallelTwoWheelVehicleModel, self).__init__(dt, nx=5, nu=2)

    def get_dynamics(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        def kinematic_jax(state_: jnp.ndarray, input_: jnp.ndarray) -> jnp.ndarray:
            vel = (input_ + state_[3:]) * 0.5
            new_state = state_[:3] + jnp.array([vel[0] * self.dt * jnp.cos(state_[2]),
                                                vel[0] * self.dt * jnp.sin(state_[2]),
                                                vel[1] * self.dt])
            new_state = jnp.concatenate([new_state, input_])
            return new_state
        return jax.jit(kinematic_jax)
    
class Legged2DModel(DynamicsBase):
    def __init__(self, dt: float, mass: float = 1.0):
        super(Legged2DModel, self).__init__(dt, nx=5, nu=2)
        self.mass = mass

    def get_dynamics(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        def legged_dynamics(state: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            # unpack state
            x, y, theta, vx, vy = state
            fx, fy = u

            # acceleration
            ax = fx / self.mass
            ay = fy / self.mass - 9.81  # gravity

            # integrate velocity and position
            vx_new = vx + ax * self.dt
            vy_new = vy + ay * self.dt
            x_new = x + vx_new * self.dt
            y_new = y + vy_new * self.dt
            theta_new = theta  # assume orientation fixed in simple planar model

            return jnp.array([x_new, y_new, theta_new, vx_new, vy_new])

        return jax.jit(legged_dynamics)

class Omni2DModel(DynamicsBase):
    def __init__(self, dt: float):
        super(Omni2DModel, self).__init__(dt, nx=3, nu=3)

    def get_dynamics(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        def omni_dynamics(state: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            x, y, theta = state
            vx_body, vy_body, omega = u

            # convert body-frame velocities to world-frame
            vx_world = vx_body * jnp.cos(theta) - vy_body * jnp.sin(theta)
            vy_world = vx_body * jnp.sin(theta) + vy_body * jnp.cos(theta)

            x_new = x + vx_world * self.dt
            y_new = y + vy_world * self.dt
            theta_new = theta + omega * self.dt

            return jnp.array([x_new, y_new, theta_new])

        return jax.jit(omni_dynamics)
    
class Marine2DModel(DynamicsBase):
    def __init__(self, dt: float, mass: float = 10.0, I_z: float = 5.0, drag_surge: float = 1.0, drag_yaw: float = 0.5):
        super(Marine2DModel, self).__init__(dt, nx=5, nu=2)
        self.mass = mass
        self.I_z = I_z
        self.drag_surge = drag_surge
        self.drag_yaw = drag_yaw

    def get_dynamics(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        def marine_dynamics(state: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            x, y, theta, v, r = state
            F_surge, tau_yaw = u

            # surge acceleration and yaw acceleration with simple linear drag
            a = (F_surge - self.drag_surge * v) / self.mass
            alpha = (tau_yaw - self.drag_yaw * r) / self.I_z

            v_new = v + a * self.dt
            r_new = r + alpha * self.dt
            theta_new = theta + r_new * self.dt
            x_new = x + v_new * jnp.cos(theta_new) * self.dt
            y_new = y + v_new * jnp.sin(theta_new) * self.dt

            return jnp.array([x_new, y_new, theta_new, v_new, r_new])

        return jax.jit(marine_dynamics)