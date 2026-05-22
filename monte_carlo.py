import random

from constants import (
    INITIAL_MASS,
    DRY_MASS,
    SAFE_LANDING_SPEED,
    DT,
)
from integrator import step
from simulation import check_landing_status, clamp_to_ground
from physics import fuel_remaining
from controller import PIDController, BangBangController
from rl_agent import QLearningController, SmoothQLearningController


METHOD = "rk4"
MAX_STEPS = 15000


def random_initial_state():
    altitude = random.uniform(600.0, 1400.0)
    velocity = random.uniform(-45.0, -5.0)
    mass = random.uniform(INITIAL_MASS * 0.9, INITIAL_MASS * 1.1)

    if mass < DRY_MASS + 50:
        mass = DRY_MASS + 50

    return [altitude, velocity, mass]


def simulate_controller(
    controller_name: str,
    initial_state: list[float],
    rl_fuel_agent=None,
    rl_smooth_agent=None,
):
    state = initial_state[:]
    time = 0.0

    pid = PIDController(kp=0.4, kd=0.5, descent_gain=0.59)
    bang_bang = BangBangController()
    if controller_name == "rl_smooth":
        rl_smooth_agent.reset()

    status = "flying"

    for _ in range(MAX_STEPS):
        if controller_name == "pid":
            throttle = pid.update(state, DT)

        elif controller_name == "bang_bang":
            throttle = bang_bang.update(state, DT)

        elif controller_name == "rl_fuel":
            throttle = rl_fuel_agent.get_throttle(state)

        elif controller_name == "rl_smooth":
            throttle = rl_smooth_agent.get_throttle(state)

        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        state = step(state, throttle, DT, METHOD)
        state = clamp_to_ground(state)
        time += DT

        status = check_landing_status(state)

        if status != "flying":
            break

    if status == "flying":
        status = "timeout"

    altitude, velocity, mass = state

    return {
        "controller": controller_name,
        "status": status,
        "time": time,
        "final_speed": abs(velocity),
        "fuel_used": initial_state[2] - mass,
        "fuel_remaining": fuel_remaining(mass),
        "safe": status == "landed" and abs(velocity) <= SAFE_LANDING_SPEED,
    }


def run_monte_carlo(trials: int = 500):
    controllers = [
        "pid",
        "bang_bang",
        "rl_fuel",
        "rl_smooth",
    ]

    rl_fuel_agent = QLearningController()
    rl_fuel_agent.load("q_table.pkl")

    rl_smooth_agent = SmoothQLearningController()
    rl_smooth_agent.load("smooth_q_table.pkl")

    results = {name: [] for name in controllers}

    for i in range(trials):
        initial_state = random_initial_state()

        for controller in controllers:
            result = simulate_controller(
                controller,
                initial_state,
                rl_fuel_agent=rl_fuel_agent,
                rl_smooth_agent=rl_smooth_agent,
            )

            results[controller].append(result)

        if i % 50 == 0:
            print(f"Completed {i}/{trials} trials")

    return results


def summarize_results(results):
    print("\nMONTE CARLO RESULTS")
    print("=" * 80)

    for controller, runs in results.items():
        total = len(runs)
        successes = sum(1 for r in runs if r["safe"])
        crashes = sum(1 for r in runs if r["status"] == "crashed")
        timeouts = sum(1 for r in runs if r["status"] == "timeout")

        success_rate = successes / total * 100
        crash_rate = crashes / total * 100
        timeout_rate = timeouts / total * 100

        successful_runs = [r for r in runs if r["safe"]]

        if successful_runs:
            avg_fuel_used = sum(r["fuel_used"] for r in successful_runs) / len(successful_runs)
            avg_final_speed = sum(r["final_speed"] for r in successful_runs) / len(successful_runs)
            avg_time = sum(r["time"] for r in successful_runs) / len(successful_runs)
        else:
            avg_fuel_used = float("nan")
            avg_final_speed = float("nan")
            avg_time = float("nan")

        print(f"\nController: {controller}")
        print(f"Success rate:      {success_rate:.2f}%")
        print(f"Crash rate:        {crash_rate:.2f}%")
        print(f"Timeout rate:      {timeout_rate:.2f}%")
        print(f"Average fuel used: {avg_fuel_used:.2f} kg")
        print(f"Average speed:     {avg_final_speed:.2f} m/s")
        print(f"Average time:      {avg_time:.2f} s")


if __name__ == "__main__":
    results = run_monte_carlo(trials=500)
    summarize_results(results)
