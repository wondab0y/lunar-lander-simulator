import random
import math
from typing import Optional

import pygame

from constants import (
    MOON_GRAVITY,
    MAX_THRUST,
    INITIAL_ALTITUDE,
    INITIAL_VELOCITY,
    INITIAL_MASS,
    DT,
)
from controller import BangBangController, PIDController
from integrator import step
from simulation import check_landing_status, clamp_to_ground
from physics import fuel_remaining


SCREEN_WIDTH = 900
SCREEN_HEIGHT = 650
GAME_WIDTH = 600
PANEL_X = 620
PANEL_WIDTH = 255
GROUND_Y = 540

FPS = 60
METHOD = "rk4"

ALTITUDE_SCALE = 0.42

LANDER_WIDTH = 42
LANDER_HEIGHT = 52

STAR_COUNT = 120
PLOT_TIME_WINDOW = 40.0
BRAKE_WARNING_DURATION = 3.0
BRAKE_WARNING_BLINKS_PER_SECOND = 3.0

TelemetrySample = tuple[float, float, float, float, float]


def reset_state() -> list[float]:
    return [INITIAL_ALTITUDE, INITIAL_VELOCITY, INITIAL_MASS]


def altitude_to_screen_y(altitude: float) -> int:
    y = GROUND_Y - altitude * ALTITUDE_SCALE
    return int(max(70, min(GROUND_Y, y)))


def draw_text(screen, font, text: str, x: int, y: int, color=(255, 255, 255)):
    image = font.render(text, True, color)
    screen.blit(image, (x, y))


def generate_stars():
    random.seed(7)
    stars = []

    for _ in range(STAR_COUNT):
        x = random.randint(0, SCREEN_WIDTH)
        y = random.randint(0, GROUND_Y - 40)
        r = random.choice([1, 1, 1, 2])
        stars.append((x, y, r))

    return stars


def draw_background(screen, stars):
    screen.fill((5, 5, 22))

    for x, y, r in stars:
        if x < GAME_WIDTH:
            pygame.draw.circle(screen, (230, 230, 230), (x, y), r)

    pygame.draw.line(
        screen,
        (140, 140, 140),
        (0, GROUND_Y + 20),
        (GAME_WIDTH, GROUND_Y + 20),
        4,
    )


def draw_lander(screen, altitude: float, throttle: float):
    x = SCREEN_WIDTH // 2
    y = altitude_to_screen_y(altitude)

    body_rect = pygame.Rect(
        x - LANDER_WIDTH // 2,
        y - LANDER_HEIGHT,
        LANDER_WIDTH,
        LANDER_HEIGHT,
    )

    pygame.draw.rect(screen, (220, 220, 220), body_rect)
    pygame.draw.rect(screen, (80, 150, 255), (x - 10, y - 42, 20, 15))

    pygame.draw.line(screen, (220, 220, 220), (x - 15, y), (x - 35, y + 20), 3)
    pygame.draw.line(screen, (220, 220, 220), (x + 15, y), (x + 35, y + 20), 3)

    if throttle > 0:
        flame_height = int(20 + 45 * throttle)
        flame_points = [
            (x - 13, y),
            (x + 13, y),
            (x, y + flame_height),
        ]
        pygame.draw.polygon(screen, (255, 140, 0), flame_points)


def scale_plot_value(value: float, min_value: float, max_value: float, rect: pygame.Rect) -> int:
    if max_value == min_value:
        return rect.centery

    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0.0, min(1.0, normalized))
    return int(rect.bottom - normalized * rect.height)


def draw_plot(
    screen,
    font,
    rect: pygame.Rect,
    title: str,
    samples: list[TelemetrySample],
    value_index: int,
    color: tuple[int, int, int],
    y_range: Optional[tuple[float, float]] = None,
):
    pygame.draw.rect(screen, (15, 18, 34), rect, border_radius=6)
    pygame.draw.rect(screen, (70, 76, 105), rect, 1, border_radius=6)

    draw_text(screen, font, title, rect.x + 8, rect.y + 6, (225, 230, 245))

    plot_rect = pygame.Rect(rect.x + 9, rect.y + 28, rect.width - 18, rect.height - 38)
    pygame.draw.line(screen, (45, 50, 74), (plot_rect.left, plot_rect.bottom), (plot_rect.right, plot_rect.bottom), 1)
    pygame.draw.line(screen, (45, 50, 74), (plot_rect.left, plot_rect.top), (plot_rect.left, plot_rect.bottom), 1)

    if len(samples) < 2:
        return

    latest_time = samples[-1][0]
    start_time = max(0.0, latest_time - PLOT_TIME_WINDOW)
    visible_samples = [sample for sample in samples if sample[0] >= start_time]

    if len(visible_samples) < 2:
        return

    if y_range is None:
        values = [sample[value_index] for sample in visible_samples]
        min_value = min(values)
        max_value = max(values)
        padding = (max_value - min_value) * 0.15 or 1.0
        min_value -= padding
        max_value += padding
    else:
        min_value, max_value = y_range

    time_span = max(PLOT_TIME_WINDOW, latest_time) if latest_time <= PLOT_TIME_WINDOW else PLOT_TIME_WINDOW
    points = []

    for sample in visible_samples:
        sample_time = sample[0]
        value = sample[value_index]
        x = plot_rect.left + int((sample_time - start_time) / time_span * plot_rect.width)
        y = scale_plot_value(value, min_value, max_value, plot_rect)
        points.append((x, y))

    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 2)


def draw_live_plots(screen, font, telemetry: list[TelemetrySample]):
    panel_rect = pygame.Rect(PANEL_X, 15, PANEL_WIDTH, 520)
    pygame.draw.rect(screen, (8, 10, 24), panel_rect, border_radius=8)
    pygame.draw.rect(screen, (55, 62, 92), panel_rect, 1, border_radius=8)

    draw_text(screen, font, "Live ODE plots", PANEL_X + 14, 28, (255, 255, 255))

    plot_width = PANEL_WIDTH - 24
    plot_height = 105
    plot_x = PANEL_X + 12
    plot_y = 62
    plot_gap = 9

    fuel_capacity = fuel_remaining(INITIAL_MASS)
    plot_specs = [
        ("Altitude (m)", 1, (80, 200, 255), (0.0, INITIAL_ALTITUDE)),
        ("Velocity (m/s)", 2, (255, 190, 85), None),
        ("Fuel (kg)", 3, (110, 255, 150), (0.0, fuel_capacity)),
        ("Throttle", 4, (255, 110, 120), (0.0, 1.0)),
    ]

    for index, (title, value_index, color, y_range) in enumerate(plot_specs):
        rect = pygame.Rect(
            plot_x,
            plot_y + index * (plot_height + plot_gap),
            plot_width,
            plot_height,
        )
        draw_plot(screen, font, rect, title, telemetry, value_index, color, y_range)


def estimate_stopping_distance(velocity: float, mass: float, has_fuel: bool) -> float:
    if velocity >= 0 or not has_fuel:
        return 0.0

    upward_acceleration = MAX_THRUST / mass - MOON_GRAVITY

    if upward_acceleration <= 0:
        return float("inf")

    return velocity * velocity / (2 * upward_acceleration)


def needs_brake_now(altitude: float, velocity: float, mass: float, has_fuel: bool) -> bool:
    stopping_distance = estimate_stopping_distance(velocity, mass, has_fuel)
    return velocity < 0 and stopping_distance >= altitude


def draw_brake_warning(screen, big_font, time: float, warning_until: float):
    if time >= warning_until:
        return

    blink_phase = math.sin(time * BRAKE_WARNING_BLINKS_PER_SECOND * math.tau)
    alpha = int(55 + 200 * (blink_phase + 1) / 2)
    scale = 1.0 + 0.04 * (blink_phase + 1) / 2

    text_image = big_font.render("BRAKE NOW", True, (255, 70, 60))
    text_image = pygame.transform.smoothscale(
        text_image,
        (
            int(text_image.get_width() * scale),
            int(text_image.get_height() * scale),
        ),
    )
    text_image.set_alpha(alpha)

    rect = text_image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text_image, rect)


def get_manual_throttle(keys, has_fuel: bool) -> float:
    if not has_fuel:
        return 0.0

    if keys[pygame.K_SPACE]:
        return 1.0

    if keys[pygame.K_DOWN]:
        return 0.75

    if keys[pygame.K_RIGHT]:
        return 0.50

    if keys[pygame.K_UP]:
        return 0.25

    return 0.0


def draw_hud(screen, font, state, throttle, status, time, ran_out_of_fuel, autopilot_mode):
    altitude, velocity, mass = state
    fuel = fuel_remaining(mass)

    velocity_color = (80, 255, 120) if abs(velocity) <= 20.0 else (255, 90, 90)
    fuel_color = (255, 90, 90) if fuel <= 0 else (255, 255, 255)

    draw_text(screen, font, f"Time: {time:.2f} s", 20, 20)
    draw_text(screen, font, f"Altitude: {altitude:.2f} m", 20, 50)
    draw_text(screen, font, f"Velocity: {velocity:.2f} m/s", 20, 80, velocity_color)
    draw_text(screen, font, f"Mass: {mass:.2f} kg", 20, 110)
    draw_text(screen, font, f"Fuel: {fuel:.2f} kg", 20, 140, fuel_color)
    draw_text(screen, font, f"Throttle: {throttle * 100:.0f}%", 20, 170)
    draw_text(screen, font, f"Autopilot: {autopilot_mode or 'OFF'}", 20, 200)
    draw_text(screen, font, f"Integrator: {METHOD}", 20, 230)
    draw_text(screen, font, f"Status: {status}", 20, 260)

    draw_text(screen, font, "UP: 25% | RIGHT: 50% | DOWN: 75% | SPACE: 100%", 20, 585)
    draw_text(screen, font, "P: PID | B: bang-bang | R: restart | ESC: quit", 20, 615)

    if velocity < -30 and status == "flying":
        draw_text(screen, font, "WARNING: DESCENDING TOO FAST", 285, 40, (255, 80, 80))

    if ran_out_of_fuel and status == "flying":
        draw_text(screen, font, "OUT OF FUEL - IMPACT EXPECTED", 285, 75, (255, 80, 80))


def main():
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lunar Lander Simulator")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    big_font = pygame.font.SysFont(None, 48)

    stars = generate_stars()

    state = reset_state()
    time = 0.0
    status = "flying"
    telemetry = [(time, state[0], state[1], fuel_remaining(state[2]), 0.0)]
    ran_out_of_fuel = False
    brake_warning_until = 0.0
    brake_warning_needed_last_frame = False
    pid_autopilot = PIDController(kp=0.4, kd=0.5, descent_gain=0.59)
    bang_bang_autopilot = BangBangController()
    autopilot_mode = None

    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_r:
                    state = reset_state()
                    time = 0.0
                    status = "flying"
                    telemetry = [(time, state[0], state[1], fuel_remaining(state[2]), 0.0)]
                    ran_out_of_fuel = False
                    brake_warning_until = 0.0
                    brake_warning_needed_last_frame = False
                    pid_autopilot.reset()
                    bang_bang_autopilot.reset()
                    autopilot_mode = None

                if event.key == pygame.K_p and status == "flying":
                    autopilot_mode = None if autopilot_mode == "PID" else "PID"
                    pid_autopilot.reset()
                    bang_bang_autopilot.reset()
                    if autopilot_mode:
                        brake_warning_until = 0.0
                        brake_warning_needed_last_frame = False

                if event.key == pygame.K_b and status == "flying":
                    autopilot_mode = None if autopilot_mode == "Bang-Bang" else "Bang-Bang"
                    pid_autopilot.reset()
                    bang_bang_autopilot.reset()
                    if autopilot_mode:
                        brake_warning_until = 0.0
                        brake_warning_needed_last_frame = False

        keys = pygame.key.get_pressed()

        altitude, velocity, mass = state
        has_fuel = fuel_remaining(mass) > 0

        if not has_fuel and altitude > 0:
            ran_out_of_fuel = True

        if autopilot_mode == "PID" and has_fuel:
            throttle = pid_autopilot.update(state, DT)
        elif autopilot_mode == "Bang-Bang" and has_fuel:
            throttle = bang_bang_autopilot.update(state, DT)
        else:
            throttle = get_manual_throttle(keys, has_fuel)

        if status == "flying":
            state = step(state, throttle, DT, METHOD)
            state = clamp_to_ground(state)
            time += DT

            status = check_landing_status(state)

            # If the spacecraft ran out of fuel before touching the ground,
            # we count the final impact as a crash.
            if status == "landed" and ran_out_of_fuel:
                status = "crashed"

        altitude, velocity, mass = state

        if telemetry[-1][0] != time:
            telemetry.append((time, altitude, velocity, fuel_remaining(mass), throttle))

        brake_warning_needed = (
            status == "flying"
            and autopilot_mode is None
            and needs_brake_now(altitude, velocity, mass, fuel_remaining(mass) > 0)
        )

        if brake_warning_needed and not brake_warning_needed_last_frame:
            brake_warning_until = time + BRAKE_WARNING_DURATION

        brake_warning_needed_last_frame = brake_warning_needed

        draw_background(screen, stars)
        draw_lander(screen, altitude, throttle)
        draw_live_plots(screen, font, telemetry)
        draw_hud(screen, font, state, throttle, status, time, ran_out_of_fuel, autopilot_mode)
        if status == "flying" and autopilot_mode is None:
            draw_brake_warning(screen, big_font, time, brake_warning_until)

        if status == "landed":
            draw_text(screen, big_font, "SAFE LANDING", 330, 295, (80, 255, 120))

        elif status == "crashed":
            draw_text(screen, big_font, "CRASH", 390, 295, (255, 80, 80))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
