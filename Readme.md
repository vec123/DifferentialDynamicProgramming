# Differential Dynamic Programming (DDP)

A high-performance implementation of Differential Dynamic Programming and Augmented Lagrangian DDP (ALDDP) for trajectory optimization and control of nonlinear dynamical systems, powered by JAX for efficient computation on CPU and GPU.

## Overview

This library provides a robust framework for solving trajectory optimization problems using Differential Dynamic Programming, a powerful backward-forward iterative algorithm for nonlinear optimal control. The implementation leverages JAX's automatic differentiation and JIT compilation for computational efficiency, making it suitable for real-time and large-scale trajectory planning applications.

**Key Capabilities:**
- Unconstrained and constrained trajectory optimization
- Support for multiple robot dynamics models (2D vehicles, marine systems, legged systems)
- Collision avoidance constraints
- Control input saturation constraints
- JAX-based differentiable computation pipeline
- Efficient JIT compilation for fast convergence

## What is Differential Dynamic Programming?

Differential Dynamic Programming (DDP) is a second-order trajectory optimization algorithm that solves the nonlinear optimal control problem by performing backward and forward sweeps through time:

- **Backward Pass:** Computes the value function approximation using local quadratic models
- **Forward Pass:** Generates an improved trajectory using the computed feedback gains
- **Augmented Lagrangian Extension:** Handles constraints through Lagrange multipliers

This approach is particularly effective for problems with:
- Nonlinear dynamics
- Nonconvex cost functions
- Complex state and control constraints

## Installation

### Prerequisites
- Python >= 3.8
- JAX and JAXLib
- Flax (for neural network utilities if needed)
- Matplotlib (for visualization)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd DifferentialDynamicProgramming
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install jax jaxlib flax matplotlib numpy
```

## Project Structure

```
DifferentialDynamicProgramming/
├── ddp/                          # Core DDP implementation
│   ├── alddp.py                 # Augmented Lagrangian DDP controller
│   └── utils/                   # Utility functions and data structures
├── dynamics/                     # Robot dynamics models
│   └── robots.py                # Implementations of various dynamics models
├── ddp_control.py               # Example control script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Supported Dynamics Models

The library includes implementations for several common robot dynamics models:

| Model | Description | Applications |
|-------|-------------|--------------|
| `TurtleModel` | 2D point robot with differential drive | Mobile robots, turtlebots |
| `ParallelTwoWheelVehicleModel` | Two parallel wheels with kinematic coupling | Wheeled vehicles |
| `Omni2DModel` | Omnidirectional 2D platform | Holonomic mobile robots |
| `Marine2DModel` | Simplified marine vessel dynamics | Autonomous surface vehicles |
| `Legged2DModel` | Simplified legged locomotion | Legged robots |

## Quick Start

### Basic Usage

```python
from ddp.alddp import ALDDPController
from dynamics.robots import TurtleModel
from ddp.utils.defines import ReferenceTrajectory, ConstraintArgs

# Initialize dynamics
dynamics = TurtleModel()

# Create controller
controller = ALDDPController(
    dynamics=dynamics,
    horizon=50,
    max_iterations=100
)

# Define reference trajectory
ref_traj = ReferenceTrajectory(...)

# Optimize trajectory
optimal_trajectory = controller.optimize(
    initial_state=x0,
    reference_trajectory=ref_traj
)
```

### Constrained Optimization

The framework supports multiple constraint types:

**Collision Avoidance:**
```python
@jax.jit
def collision_constraint(x, _, c_arg):
    obs_pos, obs_r = c_arg.obstacle.pos, c_arg.obstacle.radius
    dist = jnp.sqrt((x[0]-obs_pos[0])**2 + (x[1]-obs_pos[1])**2)
    return (obs_r + safety_margin - dist).reshape(-1)  # negative = safe
```

**Control Input Constraints:**
```python
@jax.jit
def velocity_constraint(x, u, _):
    return jnp.array([u[0] - max_velocity, -u[0]])  # Bounded velocity

@jax.jit
def angular_rate_constraint(x, u, _):
    return jnp.array([u[1] - max_omega, -max_omega - u[1]])  # Bounded angular rate
```

## Example: Turtle Robot Navigation

See `ddp_control.py` for a complete example of trajectory optimization for a turtle robot navigating around obstacles.

Run the example:
```bash
python ddp_control.py
```

This will:
1. Initialize a turtle robot with constrained dynamics
2. Define collision avoidance constraints for 3 obstacles
3. Optimize a trajectory from start to goal
4. Visualize the planned trajectory with animations

## Core Components

### ALDDPController
The main optimization engine that combines DDP with augmented Lagrangian methods to handle constraints.

**Key Methods:**
- `optimize(initial_state, reference_trajectory, constraint_args)`: Main optimization routine
- `backward_pass()`: Computes value function approximation
- `forward_pass()`: Generates improved trajectory

### ReferenceTrajectory
Defines the desired trajectory to track, including:
- Target states at each time step
- Cost weights for state tracking
- Terminal state costs

### ConstraintArgs
Container for constraint-related data:
- Obstacle positions and radii
- Input saturation limits
- Constraint penalty weights

## Advanced Features

### DDP on Lie Groups
For systems with group-theoretic structure (rotation matrices, SO(3), etc.), consider extending the algorithm with Lie-algebraic differential dynamic programming for improved numerical stability and convergence properties.

**Reference:** "Constrained Trajectory Optimization on Matrix Lie Groups via Lie-Algebraic Differential Dynamic Programming"

### Comparison with Schrödinger Bridges
The linear-quadratic case of DDP relates to the LQR-Schrödinger Bridge framework for stochastic optimal control. See "The LQR-Schrödinger Bridge" for theoretical connections.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| JAX | Latest | Automatic differentiation, JIT compilation |
| JAXLib | Latest | JAX linear algebra routines |
| Flax | Latest | Neural network utilities (optional) |
| Matplotlib | Latest | Visualization and animation |
| NumPy | Latest | Numerical computing |

## References

1. **Differential Dynamic Programming and its Application to Robot Arm Control**
   - Todorov, E., & Li, W. (2005)
   - Foundational work on DDP algorithm

2. **Trajectory Optimization on Manifolds**
   - Relevant for Lie group extensions
   - See "Constrained Trajectory Optimization on Matrix Lie Groups via Lie-Algebraic Differential Dynamic Programming"

3. **Augmented Lagrangian Methods for Constrained Optimization**
   - Combines DDP with constraint handling

## Performance Characteristics

- **Computational Speed:** JIT-compiled operations enable real-time optimization
- **Memory Efficiency:** JAX's functional paradigm minimizes memory overhead
- **Scalability:** Suitable for horizons up to 500+ timesteps on modern hardware
- **Accuracy:** Second-order convergence properties ensure precise optimal solutions

## Future Extensions

- [ ] Distributed DDP for large-scale problems
- [ ] Lie-algebraic formulation for group-theoretic systems
- [ ] Neural network-based value function approximation
- [ ] Stochastic variants for uncertainty handling
- [ ] GPU acceleration benchmarking
- [ ] ROS integration for real-world deployment

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- New features include docstrings and examples
- Tests are provided for new functionality

## License

[Specify your license here]

## Contact

For questions and collaboration, please reach out to the project maintainers.

---

**Last Updated:** May 2026
