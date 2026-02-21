import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import functools
import time
from dynamics.robots import ParallelTwoWheelVehicleModel, TurtleModel, Marine2DModel
from dynamics.robots import Omni2DModel, Legged2DModel, Marine2DModel
from dynamics.robots import DynamicsBase
from ddp.alddp import ALDDPController
from ddp.utils.defines import ReferenceTrajectory, ConstraintArgs, ObstacleData
from typing import Dict, List, Callable, Tuple, Any

# ----------------------------
# Collision Constraints
# ----------------------------
@jax.jit
def collision_constraint1(x, _, c_arg: ConstraintArgs):
    obs_pos, obs_r = c_arg.obs_data1.obs_pos, c_arg.obs_data1.obs_r
    dist = jnp.sqrt((x[0]-obs_pos[0])**2 + (x[1]-obs_pos[1])**2)
    return (obs_r + 0.3 - dist).reshape(-1)  # negative = safe

@jax.jit
def collision_constraint2(x, _, c_arg: ConstraintArgs):
    obs_pos, obs_r = c_arg.obs_data2.obs_pos, c_arg.obs_data2.obs_r
    dist = jnp.sqrt((x[0]-obs_pos[0])**2 + (x[1]-obs_pos[1])**2)
    return (obs_r + 0.3 - dist).reshape(-1)

@jax.jit
def collision_constraint3(x, _, c_arg: ConstraintArgs):
    obs_pos, obs_r = c_arg.obs_data3.obs_pos, c_arg.obs_data3.obs_r
    dist = jnp.sqrt((x[0]-obs_pos[0])**2 + (x[1]-obs_pos[1])**2)
    return (obs_r + 0.3 - dist).reshape(-1)

# ----------------------------
# Input Constraints
# ----------------------------
@jax.jit
def v_constraint(x, u, _):
    return jnp.array([u[0]-2.0, 0.0-u[0]])

@jax.jit
def omega_constraint(x, u, _):
    return jnp.array([u[1]-jnp.pi/2, -jnp.pi/2-u[1]])

# ----------------------------
# Main
# ----------------------------
def main():
    outer_iters = 100
    iterations = 30
    robot_model: DynamicsBase = Marine2DModel(dt=0.1)
    dynamics: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = robot_model.get_dynamics()

    nx = robot_model.nx  # state dimension from model
    nu = robot_model.nu  # control dimension from model
    horizon = 20         # can still be fixed or parameterized

    # Controller weights
    Q = jnp.eye(nx)
    Q_terminal = jnp.eye(nx) * 800  
    #Q = jnp.diag(jnp.concatenate([jnp.ones(3), jnp.zeros(nx-3)]))  
    #Q_terminal = jnp.diag(jnp.concatenate([jnp.ones(3)*800, jnp.zeros(nx-3)]))

    R = jnp.eye(nu)  

    # Start & goal
    start = jnp.zeros(nx)
    goal  = jnp.zeros(nx)

    # Fill only meaningful states (x, y, theta)
    start = start.at[:3].set(jnp.array([1.0, 1.0, 0.0]))
    goal  = goal.at[:3].set(jnp.array([10.0, 10.0, 0.0]))
    state = start

    x_vals = jnp.linspace(start[0], goal[0], horizon+1)
    y_vals = jnp.linspace(start[1], goal[1], horizon+1)
    theta_vals = jnp.linspace(start[2], goal[2], horizon+1)



    # Obstacles
    c_args = ConstraintArgs(
        obs_data1=ObstacleData(
            obs_pos=jnp.repeat(jnp.array([[3.5, 2.0, 0.0]]), horizon, axis=0),  # shape (horizon,3)
            obs_r=jnp.repeat(jnp.array([[1.0]]), horizon, axis=0)               # shape (horizon,1)
        ),
        obs_data2=ObstacleData(
            obs_pos=jnp.repeat(jnp.array([[6.5, 7.0, 0.0]]), horizon, axis=0),
            obs_r=jnp.repeat(jnp.array([[1.0]]), horizon, axis=0)
        ),
        obs_data3=ObstacleData(
            obs_pos=jnp.repeat(jnp.array([[5.5, 5.5, 0.0]]), horizon, axis=0),
            obs_r=jnp.repeat(jnp.array([[1.0]]), horizon, axis=0)
        )
    )
        # Constraints
    f_constraints = [collision_constraint1, collision_constraint2, collision_constraint3]

    ref_traj = ReferenceTrajectory(
        ref_xs=jnp.zeros((horizon+1, nx)),
        ref_us=jnp.ones((horizon, nu)),
        #ref_xs = jnp.stack([x_vals, y_vals, theta_vals], axis=1), 
        #ref_us = jnp.zeros((horizon, nu)),
        lambdas=jnp.zeros((horizon, len(f_constraints)))  # total constraint values
    )

    print("ref_traj.ref_xs: ", ref_traj.ref_xs)


    # Controller
    controller = ALDDPController(
        f_dynamics=dynamics,
        f_constraints=f_constraints,
        Q=Q,
        Q_terminal=Q_terminal,
        R=R,
        horizon=horizon,
        max_iter = iterations,
    )

    # JIT compile
    r, l = controller.calc_input(state, goal, c_args, ref_traj)
    print("r:", r)
    print("l:", l)

    # Rollout
    states, inputs, predicted_trajs = [], [], []
    start_time = time.perf_counter()
    for _ in range(outer_iters):
        u, ref_traj = controller.calc_input(state, goal, c_args, ref_traj)
        state = dynamics(state, u)
        states.append(np.array(state))
        inputs.append(np.array(u))
        predicted_trajs.append(np.array(ref_traj.ref_xs))
    print(f"Elapsed time: {time.perf_counter()-start_time:.2f}s")

    states = np.array(states)
    inputs = np.array(inputs)

    # Save trajectory plot
    print("states.shape: ", states.shape)
    print("inputs.shape: ", inputs.shape)
    print(states[:,0])
    print(inputs[:,0])
    fig, ax = plt.subplots()
    ax.plot(states[:,0], states[:,1], 'b-', label='trajectory')
    ax.plot(start[0], start[1], 'go', label='start')
    ax.plot(goal[0], goal[1], 'ro', label='goal')
    # plot obstacles
    for obs in [c_args.obs_data1, c_args.obs_data2, c_args.obs_data3]:
        # Take the first timestep's x, y position and convert to Python floats
        x, y = float(obs.obs_pos[0, 0]), float(obs.obs_pos[0, 1])
        radius = float(obs.obs_r[0, 0])  # convert 1D JAX array to Python float
        circle = plt.Circle((x, y), radius, color='red', alpha=0.3)
        ax.add_patch(circle)


    ax.set_aspect('equal')
    ax.legend()
    plt.savefig("turtle_alddp_fixed.png")
    plt.close()

if __name__ == "__main__":
    main()
