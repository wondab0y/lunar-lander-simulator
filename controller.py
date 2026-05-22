import math

from constants import (
    DRY_MASS,
    EXHAUST_VELOCITY,
    MAX_THRUST,
    MOON_GRAVITY,
)


class PIDController:
    def __init__(self, kp=0.08, ki=0.0, kd=0.08, descent_gain=1.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.descent_gain = descent_gain

        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    def target_velocity(self, altitude: float) -> float:
        altitude = max(altitude, 0.0)
        return -self.descent_gain * math.sqrt(altitude)

    def update(self, state: list[float], dt: float) -> float:
        altitude, velocity, mass = state

        v_target = self.target_velocity(altitude)

        error = v_target - velocity

        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        desired_acceleration = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        required_thrust = mass * (MOON_GRAVITY + desired_acceleration)

        throttle = required_thrust / MAX_THRUST

        return max(0.0, min(1.0, throttle))


class BangBangController:
    def __init__(self):
        self.burning = False

    def reset(self):
        self.burning = False

    def stopping_distance(self, velocity: float, mass: float) -> float:
        if velocity >= 0:
            return 0.0

        acceleration = MAX_THRUST / mass - MOON_GRAVITY

        if acceleration <= 0:
            return float("inf")

        return velocity * velocity / (2 * acceleration)

    def update(self, state: list[float], dt: float) -> float:
        if self.burning:
            return 1.0

        if self.must_burn_now(state):
            self.burning = True
            return 1.0

        return 0.0

    def burn_time_until_empty(self, mass: float) -> float:
        fuel = max(0.0, mass - DRY_MASS)
        return fuel * EXHAUST_VELOCITY / MAX_THRUST

    def mass_after_burn(self, mass: float, burn_time: float) -> float:
        mass_flow = MAX_THRUST / EXHAUST_VELOCITY
        return max(DRY_MASS, mass - mass_flow * burn_time)

    def velocity_after_burn(self, velocity: float, mass: float, burn_time: float) -> float:
        remaining_mass = self.mass_after_burn(mass, burn_time)
        return (
            velocity
            + EXHAUST_VELOCITY * math.log(mass / remaining_mass)
            - MOON_GRAVITY * burn_time
        )

    def altitude_after_burn(
        self,
        altitude: float,
        velocity: float,
        mass: float,
        burn_time: float,
    ) -> float:
        mass_flow = MAX_THRUST / EXHAUST_VELOCITY
        remaining_mass = self.mass_after_burn(mass, burn_time)
        thrust_integral = (
            mass
            - remaining_mass * (math.log(mass / remaining_mass) + 1.0)
        ) / mass_flow

        return (
            altitude
            + velocity * burn_time
            + EXHAUST_VELOCITY * thrust_integral
            - 0.5 * MOON_GRAVITY * burn_time * burn_time
        )

    def must_burn_now(self, state: list[float]) -> bool:
        altitude, velocity, mass = state

        if velocity >= 0 or mass <= DRY_MASS:
            return False

        max_burn_time = self.burn_time_until_empty(mass)

        if max_burn_time <= 0:
            return False

        if self.velocity_after_burn(velocity, mass, max_burn_time) < 0:
            return True

        low = 0.0
        high = max_burn_time

        for _ in range(40):
            mid = (low + high) / 2.0
            if self.velocity_after_burn(velocity, mass, mid) < 0:
                low = mid
            else:
                high = mid

        stop_altitude = self.altitude_after_burn(altitude, velocity, mass, high)

        return stop_altitude <= 0.0
