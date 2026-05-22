import math

from constants import MAX_THRUST, MOON_GRAVITY


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
    def __init__(self, safety_margin=0.95):
        self.safety_margin = safety_margin
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
        altitude, velocity, mass = state

        if self.burning:
            return 1.0

        if self.stopping_distance(velocity, mass) * self.safety_margin >= altitude:
            self.burning = True
            return 1.0

        return 0.0
