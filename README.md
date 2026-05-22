# Lunar Lander Simulator

A 1D lunar landing simulation project for differential equations.

The project models vertical spacecraft motion under lunar gravity with thrust, fuel consumption, and changing spacecraft mass.

## Features

- Variable-mass spacecraft dynamics
- Lunar gravity model
- RK4 numerical integration
- Manual thrust control
- Fuel consumption
- Safe landing / crash detection
- Plot generation for altitude, velocity, and fuel

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Controls

- `SPACE` — apply thrust
- `R` — restart
- `P` — toggle autopilot

## Planned Extensions

- PID autopilot
- Controller optimization
- Monte Carlo robustness testing
- Reinforcement learning controller
