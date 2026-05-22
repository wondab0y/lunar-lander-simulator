"""
rl_agent.py

Simple tabular Q-learning controller for the lunar lander.

The agent learns a policy:

    state -> throttle action

State is discretized from:
    altitude
    velocity
    fuel

Actions are discrete throttle levels:
    0%, 25%, 50%, 75%, 100%
"""

import random
import pickle
import sys

from constants import (
    MAX_THRUST,
    MOON_GRAVITY,
    INITIAL_ALTITUDE,
    INITIAL_VELOCITY,
    INITIAL_MASS,
    SAFE_LANDING_SPEED,
    DT,
)
from integrator import step
from simulation import check_landing_status, clamp_to_ground
from physics import fuel_remaining


ACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
MIN_POLICY_CONFIDENCE = 1.0
SMOOTH_Q_TABLE_FILE = "smooth_q_table.pkl"
MAX_TRAINING_ALTITUDE = INITIAL_ALTITUDE + 40.0
SMOOTH_FRAME_PENALTY = 0.35
SMOOTH_FUEL_PENALTY = 0.20
SMOOTH_SPEED_PENALTY = 0.01
SMOOTHNESS_PENALTY = 0.75


class QLearningController:
    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.995,
        epsilon: float = 0.35,
    ):
        self.q_table = {}

        self.alpha = alpha          # learning rate
        self.gamma = gamma          # future reward discount
        self.epsilon = epsilon      # exploration probability

    def discretize_state(self, state: list[float]) -> tuple[int, int, int]:
        altitude, velocity, mass = state
        fuel = fuel_remaining(mass)

        altitude_bin = int(max(0, min(1200, altitude)) // 20)
        velocity_bin = int(max(-120, min(40, velocity)) // 4)
        fuel_bin = int(max(0, min(60, fuel)) // 10)

        return altitude_bin, velocity_bin, fuel_bin

    def get_q_values(self, discrete_state):
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0 for _ in ACTIONS]

        return self.q_table[discrete_state]

    def fallback_throttle(self, state: list[float]) -> float:
        altitude, velocity, mass = state

        if velocity >= 0:
            return 0.0

        braking_acceleration = MAX_THRUST / mass - MOON_GRAVITY

        if braking_acceleration <= 0:
            return 1.0

        stopping_distance = velocity * velocity / (2 * braking_acceleration)

        if stopping_distance >= altitude * 0.95:
            return 1.0

        return 0.0

    def choose_action_index(self, state: list[float], training: bool = True) -> int:
        discrete_state = self.discretize_state(state)
        q_values = self.get_q_values(discrete_state)

        if training and random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))

        return max(range(len(ACTIONS)), key=lambda i: q_values[i])

    def get_throttle(self, state: list[float]) -> float:
        discrete_state = self.discretize_state(state)
        q_values = self.get_q_values(discrete_state)

        if max(q_values) - min(q_values) < MIN_POLICY_CONFIDENCE:
            return self.fallback_throttle(state)

        action_index = max(range(len(ACTIONS)), key=lambda i: q_values[i])
        return ACTIONS[action_index]

    def update(self, state, action_index, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        next_discrete_state = self.discretize_state(next_state)

        q_values = self.get_q_values(discrete_state)
        next_q_values = self.get_q_values(next_discrete_state)

        old_q = q_values[action_index]

        if done:
            target = reward
        else:
            target = reward + self.gamma * max(next_q_values)

        q_values[action_index] = old_q + self.alpha * (target - old_q)

    def save(self, filename: str = "q_table.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(self.q_table, file)

    def load(self, filename: str = "q_table.pkl"):
        with open(filename, "rb") as file:
            self.q_table = pickle.load(file)


class SmoothQLearningController(QLearningController):
    def __init__(
        self,
        alpha: float = 0.12,
        gamma: float = 0.995,
        epsilon: float = 0.12,
    ):
        super().__init__(alpha=alpha, gamma=gamma, epsilon=epsilon)
        self.previous_throttle = 0.0

    def reset(self):
        self.previous_throttle = 0.0

    def fallback_throttle(self, state: list[float]) -> float:
        target_throttle = super().fallback_throttle(state)

        if target_throttle > self.previous_throttle:
            return min(1.0, self.previous_throttle + 0.25)

        if target_throttle < self.previous_throttle:
            return max(0.0, self.previous_throttle - 0.25)

        return target_throttle

    def discretize_state_with_previous_throttle(
        self,
        state: list[float],
        previous_throttle: float,
    ) -> tuple[int, int, int, int]:
        altitude_bin, velocity_bin, fuel_bin = self.discretize_state(state)
        throttle_bin = min(
            range(len(ACTIONS)),
            key=lambda index: abs(ACTIONS[index] - previous_throttle),
        )

        return altitude_bin, velocity_bin, fuel_bin, throttle_bin

    def choose_action_index(
        self,
        state: list[float],
        previous_throttle: float = None,
        training: bool = True,
    ) -> int:
        if previous_throttle is None:
            previous_throttle = self.previous_throttle

        discrete_state = self.discretize_state_with_previous_throttle(
            state,
            previous_throttle,
        )
        q_values = self.get_q_values(discrete_state)

        if training and random.random() < self.epsilon:
            throttle_index = discrete_state[-1]
            low = max(0, throttle_index - 1)
            high = min(len(ACTIONS) - 1, throttle_index + 1)
            return random.randint(low, high)

        if max(q_values) - min(q_values) < MIN_POLICY_CONFIDENCE:
            saved_previous_throttle = self.previous_throttle
            self.previous_throttle = previous_throttle
            fallback_throttle = self.fallback_throttle(state)
            self.previous_throttle = saved_previous_throttle
            return min(
                range(len(ACTIONS)),
                key=lambda index: abs(ACTIONS[index] - fallback_throttle),
            )

        return max(range(len(ACTIONS)), key=lambda i: q_values[i])

    def get_throttle(self, state: list[float]) -> float:
        discrete_state = self.discretize_state_with_previous_throttle(
            state,
            self.previous_throttle,
        )
        q_values = self.get_q_values(discrete_state)

        if max(q_values) - min(q_values) < MIN_POLICY_CONFIDENCE:
            throttle = self.fallback_throttle(state)
        else:
            action_index = max(range(len(ACTIONS)), key=lambda i: q_values[i])
            throttle = ACTIONS[action_index]

        safety_throttle = self.fallback_throttle(state)
        if safety_throttle > throttle:
            throttle = safety_throttle

        self.previous_throttle = throttle
        return throttle

    def update(
        self,
        state,
        previous_throttle,
        action_index,
        reward,
        next_state,
        done,
    ):
        throttle = ACTIONS[action_index]
        discrete_state = self.discretize_state_with_previous_throttle(
            state,
            previous_throttle,
        )
        next_discrete_state = self.discretize_state_with_previous_throttle(
            next_state,
            throttle,
        )

        q_values = self.get_q_values(discrete_state)
        next_q_values = self.get_q_values(next_discrete_state)

        old_q = q_values[action_index]

        if done:
            target = reward
        else:
            target = reward + self.gamma * max(next_q_values)

        q_values[action_index] = old_q + self.alpha * (target - old_q)


def compute_regular_reward(state, throttle, status):
    altitude, velocity, mass = state
    fuel = fuel_remaining(mass)

    fuel_penalty = 0.08 * throttle
    speed_penalty = 0.05 * abs(velocity)
    target_descent_speed = min(40.0, 0.7 * altitude ** 0.5)
    descent_error = abs(abs(velocity) - target_descent_speed)

    if status == "landed":
        landing_speed = abs(velocity)

        if landing_speed <= SAFE_LANDING_SPEED:
            return 2000.0 - 60.0 * landing_speed + 8.0 * fuel

        return -2000.0 - 100.0 * landing_speed

    if status == "crashed":
        return -2500.0 - 80.0 * abs(velocity)

    return -0.15 - fuel_penalty - speed_penalty - 0.02 * descent_error


def compute_reward(state, throttle, previous_throttle, status, smooth=False):
    altitude, velocity, mass = state

    if smooth:
        frame_penalty = SMOOTH_FRAME_PENALTY
        fuel_penalty = SMOOTH_FUEL_PENALTY * throttle
        speed_penalty = SMOOTH_SPEED_PENALTY * abs(velocity)
        smoothness_penalty = SMOOTHNESS_PENALTY * abs(throttle - previous_throttle)
    else:
        frame_penalty = 1.0
        fuel_penalty = 0.5 * throttle
        speed_penalty = 0.02 * abs(velocity)
        smoothness_penalty = 0.0


    if status == "landed":
        landing_speed = abs(velocity)
        return 1000.0 - 10.0 * landing_speed

    if status == "crashed":
        return -1000.0 - abs(velocity)

    return -frame_penalty - fuel_penalty - speed_penalty - smoothness_penalty


def train_q_learning(
    episodes: int = 12000,
    max_steps: int = 4500,
    method: str = "rk4",
):
    agent = QLearningController()

    for episode in range(episodes):
        state = [
            INITIAL_ALTITUDE,
            INITIAL_VELOCITY,
            INITIAL_MASS,
        ]

        status = "flying"
        total_reward = 0.0

        for _ in range(max_steps):
            action_index = agent.choose_action_index(state, training=True)
            throttle = ACTIONS[action_index]

            next_state = step(state, throttle, DT, method)
            next_state = clamp_to_ground(next_state)

            status = check_landing_status(next_state)
            done = status != "flying"

            reward = compute_regular_reward(next_state, throttle, status)

            agent.update(
                state,
                action_index,
                reward,
                next_state,
                done,
            )

            state = next_state
            total_reward += reward

            if done:
                break

        agent.epsilon = max(0.03, agent.epsilon * 0.9995)

        if episode % 250 == 0:
            print(
                f"Episode {episode:5d} | "
                f"status={status:8s} | "
                f"reward={total_reward:9.2f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    agent.save()
    print("Training finished. Saved q_table.pkl")

    return agent


def train_smooth_q_learning(
    episodes: int = 12000,
    max_steps: int = 2200,
    method: str = "rk4",
):
    agent = SmoothQLearningController()

    for episode in range(episodes):
        state = [
            INITIAL_ALTITUDE,
            INITIAL_VELOCITY,
            INITIAL_MASS,
        ]

        status = "flying"
        total_reward = 0.0
        previous_throttle = 0.0

        for _ in range(max_steps):
            action_index = agent.choose_action_index(
                state,
                previous_throttle=previous_throttle,
                training=True,
            )
            throttle = ACTIONS[action_index]

            next_state = step(state, throttle, DT, method)
            next_state = clamp_to_ground(next_state)

            status = check_landing_status(next_state)
            if next_state[0] > MAX_TRAINING_ALTITUDE:
                status = "crashed"
            done = status != "flying"

            reward = compute_reward(
                next_state,
                throttle,
                previous_throttle,
                status,
                smooth=True,
            )

            agent.update(
                state,
                previous_throttle,
                action_index,
                reward,
                next_state,
                done,
            )

            state = next_state
            previous_throttle = throttle
            total_reward += reward

            if done:
                break

        agent.epsilon = max(0.02, agent.epsilon * 0.9993)

        if episode % 250 == 0:
            print(
                f"Smooth episode {episode:5d} | "
                f"status={status:8s} | "
                f"reward={total_reward:9.2f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    agent.save(SMOOTH_Q_TABLE_FILE)
    print(f"Smooth training finished. Saved {SMOOTH_Q_TABLE_FILE}")

    return agent


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smooth":
        train_smooth_q_learning()
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        train_q_learning()
        train_smooth_q_learning()
    else:
        train_q_learning()
