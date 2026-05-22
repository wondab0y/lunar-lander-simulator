from constants import SAFE_LANDING_SPEED, DRY_MASS


def check_landing_status(state: list[float]) -> str:
    altitude, velocity, mass = state

    if altitude > 0:
        return "flying"

    if abs(velocity) <= SAFE_LANDING_SPEED:
        return "landed"

    return "crashed"


def clamp_to_ground(state: list[float]) -> list[float]:
    altitude, velocity, mass = state

    if altitude < 0:
        altitude = 0.0

    return [altitude, velocity, mass]


def is_out_of_fuel(state: list[float]) -> bool:
    return state[2] <= DRY_MASS