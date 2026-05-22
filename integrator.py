"""
integrator.py

Numerical integration methods for the lunar lander ODE system.

Implemented methods:
    - Euler method
    - Midpoint method / RK2
    - Runge-Kutta 4th order / RK4
"""

from physics import lander_derivatives


def add_scaled(state: list[float], derivatives: list[float], scale: float) -> list[float]:
    return [
        state[i] + scale * derivatives[i]
        for i in range(len(state))
    ]


def euler_step(state: list[float], throttle: float, dt: float) -> list[float]:
    """
    one time step advancement for explicit Euler method.

    formula:

        y_{n+1} = y_n + dt * f(y_n)

    the most inaccurate method out of the 3.
    """

    derivatives = lander_derivatives(state, throttle)

    return add_scaled(state, derivatives, dt)


def midpoint_step(state: list[float], throttle: float, dt: float) -> list[float]:
    """
    RK2 method.

    formula:

        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        y_{n+1} = y_n + dt * k2

    improves Euler.
    """

    k1 = lander_derivatives(state, throttle)
    midpoint_state = add_scaled(state, k1, dt / 2)

    k2 = lander_derivatives(midpoint_state, throttle)

    return add_scaled(state, k2, dt)


def rk4_step(state: list[float], throttle: float, dt: float) -> list[float]:
    """
    RK4.

    formula:

        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        k3 = f(y_n + dt/2 * k2)
        k4 = f(y_n + dt * k3)

        y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)

    the most accurate.
    """

    k1 = lander_derivatives(state, throttle)

    k2_state = add_scaled(state, k1, dt / 2)
    k2 = lander_derivatives(k2_state, throttle)

    k3_state = add_scaled(state, k2, dt / 2)
    k3 = lander_derivatives(k3_state, throttle)

    k4_state = add_scaled(state, k3, dt)
    k4 = lander_derivatives(k4_state, throttle)

    new_state = [
        state[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(len(state))
    ]

    return new_state


def step(state: list[float], throttle: float, dt: float, method: str = "rk4") -> list[float]:

    if method == "euler":
        return euler_step(state, throttle, dt)

    if method == "midpoint":
        return midpoint_step(state, throttle, dt)

    if method == "rk4":
        return rk4_step(state, throttle, dt)

    raise ValueError(f"Unknown integration method: {method}")