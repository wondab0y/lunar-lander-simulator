# Lunar Lander Simulator

A physics-based lunar landing simulator implemented in Python.

The project models powered descent under nonlinear variable-mass rocket dynamics and compares several landing strategies:

- Manual control
- PID autopilot
- Bang-bang autopilot
- Reinforcement learning controllers using Q-learning

The simulator also includes numerical integration methods, live telemetry, controller comparison tools, and Monte Carlo experiments.

## Features

### Physics Simulation

- One-dimensional lunar landing model
- Variable-mass rocket dynamics
- Moon gravity
- Fuel consumption
- Throttle-based thrust control

### Numerical Methods

- Explicit Euler method
- Midpoint method
- Fourth-order Runge-Kutta method (RK4)

### Controllers

- Manual control
- PID controller
- Bang-bang controller based on variable-mass braking equations
- Fuel-optimized Q-learning controller
- Smooth-control Q-learning controller

### Experiments

- Monte Carlo robustness testing
- Controller performance comparison
- Offline plots using Jupyter notebooks
- Phase-space and trajectory analysis

## Installation

Clone the repository:

```bash
git clone https://github.com/wondab0y/lunar-lander-simulator.git
cd lunar-lander-simulator
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Game

```bash
python3 main.py
```

On Windows, use:

```bash
python main.py
```

## Controls

### Manual Control

| Key | Action |
|---|---|
| Up Arrow | 25% throttle |
| Right Arrow | 50% throttle |
| Down Arrow | 75% throttle |
| Space | 100% throttle |
| R | Restart |
| Esc | Quit |

### Autopilot Modes

| Key | Controller |
|---|---|
| P | PID autopilot |
| B | Bang-bang autopilot |
| F | Fuel-optimized RL autopilot |
| S | Smooth-control RL autopilot |

## Reinforcement Learning

Train the Q-learning agents:

```bash
python3 rl_agent.py
```

This generates:

```text
q_table.pkl
smooth_q_table.pkl
```

These files are used by the reinforcement learning autopilot modes.

## Monte Carlo Experiments

Run statistical controller comparison:

```bash
python3 monte_carlo.py
```

The Monte Carlo script compares the controllers using randomized initial conditions and reports metrics such as:

- success rate
- crash rate
- average fuel consumption
- average landing speed
- average landing time

## Jupyter Experiments

Open the notebook:

```bash
jupyter notebook
```

Then run:

```text
experiments.ipynb
```

The notebook is used for offline analysis and plot generation.

## Project Structure

```text
lunar-lander-simulator/
│
├── main.py
├── physics.py
├── integrator.py
├── controller.py
├── rl_agent.py
├── monte_carlo.py
├── simulation.py
├── constants.py
│
├── q_table.pkl
├── smooth_q_table.pkl
│
├── experiments.ipynb
├── requirements.txt
└── README.md
```

## Mathematical Background

The simulator is based on the ODE system:

```text
dh/dt = v
dv/dt = T/m - g
dm/dt = -T/v_e
```

where:

- `h` is altitude
- `v` is vertical velocity
- `m` is spacecraft mass
- `T` is thrust
- `g` is lunar gravity
- `v_e` is effective exhaust velocity

The project combines ideas from:

- ordinary differential equations
- numerical integration
- rocket dynamics
- optimal control
- PID feedback control
- reinforcement learning
- Monte Carlo simulation

## Author

Yehor Pantelieiev

GitHub: https://github.com/wondab0y/lunar-lander-simulator
