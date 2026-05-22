"""
physics.py

Physical model for a 1D lunar lander.

State vector:
    state[0] = altitude h(t), meters
    state[1] = velocity v(t), meters / second
    state[2] = mass     m(t), kilograms

Sign convention:
    Positive altitude - above the Moon surface.
    Positive velocity - moving upward.
    Negative velocity - falling downward.

Control:
    throttle u(t), where 0 <= u <= 1

    u = 0 - engine off.
    u = 1 - max thrust.

Equations:
    dh/dt = v
    dv/dt = T / m - g
    dm/dt = -T / v_e

where:
    T = u * T_max
    g = lunar gravity
    v_e = effective exhaust velocity
"""

from constants import MOON_GRAVITY, MAX_THRUST, EXHAUST_VELOCITY, DRY_MASS


def clamp(value: float, low: float, high: float) -> float:
    # make value belong to [low, high] interval

    if value < low:
        return low

    if value > high:
        return high

    return value


def clamp_throttle(throttle: float, mass: float) -> float:
    """
    make sure values are valid : 
    1. spacecraft reaching dry mass means there is no fuel left => engine must be turned off
    2. throttle is normalized thrust => stays in [0, 1]
    """

    if mass <= DRY_MASS:
        return 0.0

    return clamp(throttle, 0.0, 1.0)


def throttle_to_thrust(throttle: float, mass: float) -> float:
    throttle = clamp_throttle(throttle, mass)
    return throttle * MAX_THRUST


def lander_derivatives(state: list[float], throttle: float) -> list[float]:
    """
    compute the derivatives of the lander's state derived from equations.
    
    parameters:
        state:
            [altitude, velocity, mass]

        throttle:
            engine throttle in range [0, 1].

    return:
        [dh_dt, dv_dt, dm_dt]
    """

    altitude, velocity, mass = state

    if mass <= 0:
        raise ValueError("Mass must be positive.")

    thrust = throttle_to_thrust(throttle, mass)

    dh_dt = velocity

    # acceleration = upward thrust acceleration - downward gravity
    dv_dt = thrust / mass - MOON_GRAVITY

    # thrust = exhaust_velocity * fuel_mass_flow_rate (derived from 2nd Newton's law for impulses)
    # total mass decreases by thrust / exhaust_velocity.
    dm_dt = -thrust / EXHAUST_VELOCITY

    if mass <= DRY_MASS and dm_dt < 0:
        dm_dt = 0.0

    return [dh_dt, dv_dt, dm_dt]


def is_out_of_fuel(mass: float) -> bool:
    return mass <= DRY_MASS


def fuel_remaining(mass: float) -> float:
    return max(0.0, mass - DRY_MASS)