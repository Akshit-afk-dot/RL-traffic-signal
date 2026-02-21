import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pygame


WINDOW_SIZE = 800
FPS = 60
ROAD_WIDTH = 220
CAR_LENGTH = 28
CAR_WIDTH = 14
CAR_SPEED = 2.2
SPAWN_INTERVAL_MS = 1500
LIGHT_SWITCH_MS = 5000


@dataclass
class Road:
    """A rectangular road segment."""

    x: int
    y: int
    width: int
    height: int
    color: Tuple[int, int, int] = (55, 55, 55)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))


class TrafficLight:
    """A simple binary traffic light."""

    RED = "RED"
    GREEN = "GREEN"

    def __init__(self, initial_state: str = RED, switch_interval_ms: int = LIGHT_SWITCH_MS) -> None:
        self.state = initial_state
        self.switch_interval_ms = switch_interval_ms
        self._last_switch_time_ms = pygame.time.get_ticks()

    def update(self, now_ms: int) -> None:
        if now_ms - self._last_switch_time_ms >= self.switch_interval_ms:
            self.state = self.GREEN if self.state == self.RED else self.RED
            self._last_switch_time_ms = now_ms

    def get_state(self) -> str:
        return self.state


class Car:
    """Vehicle that moves straight through one lane."""

    _LANE_CONFIG: Dict[str, Dict[str, Tuple[float, float]]] = {
        "N": {"dir": (0.0, 1.0), "spawn": (WINDOW_SIZE / 2 - 25, -CAR_LENGTH)},
        "S": {"dir": (0.0, -1.0), "spawn": (WINDOW_SIZE / 2 + 11, WINDOW_SIZE + CAR_LENGTH)},
        "W": {"dir": (1.0, 0.0), "spawn": (-CAR_LENGTH, WINDOW_SIZE / 2 + 11)},
        "E": {"dir": (-1.0, 0.0), "spawn": (WINDOW_SIZE + CAR_LENGTH, WINDOW_SIZE / 2 - 25)},
    }

    def __init__(self, lane: str, speed: float = CAR_SPEED) -> None:
        if lane not in self._LANE_CONFIG:
            raise ValueError(f"Unsupported lane: {lane}")

        self.lane = lane
        self.speed = speed
        self.position = list(self._LANE_CONFIG[lane]["spawn"])
        self.direction = self._LANE_CONFIG[lane]["dir"]

        if lane in {"N", "S"}:
            self.size = (CAR_WIDTH, CAR_LENGTH)
        else:
            self.size = (CAR_LENGTH, CAR_WIDTH)

    def _is_before_stop_line(self, intersection_rect: pygame.Rect) -> bool:
        x, y = self.position
        if self.lane == "N":
            front = y + self.size[1]
            return front <= intersection_rect.top
        if self.lane == "S":
            front = y
            return front >= intersection_rect.bottom
        if self.lane == "W":
            front = x + self.size[0]
            return front <= intersection_rect.left
        front = x
        return front >= intersection_rect.right

    def update(self, light_state: str, intersection_rect: pygame.Rect) -> None:
        should_stop = light_state == TrafficLight.RED and self._is_before_stop_line(intersection_rect)
        if should_stop:
            return

        self.position[0] += self.direction[0] * self.speed
        self.position[1] += self.direction[1] * self.speed

    def is_out_of_bounds(self, width: int, height: int) -> bool:
        x, y = self.position
        return x < -100 or x > width + 100 or y < -100 or y > height + 100

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, (30, 144, 255), (*self.position, *self.size))


class TrafficEnv:
    """Simulation environment decoupled into update and render passes."""

    def __init__(self, width: int = WINDOW_SIZE, height: int = WINDOW_SIZE) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MARL Traffic Intersection - Phase 1")
        self.clock = pygame.time.Clock()

        self.running = True
        self.last_spawn_ms = pygame.time.get_ticks()

        self.vertical_road = Road(self.width // 2 - ROAD_WIDTH // 2, 0, ROAD_WIDTH, self.height)
        self.horizontal_road = Road(0, self.height // 2 - ROAD_WIDTH // 2, self.width, ROAD_WIDTH)

        self.intersection_rect = pygame.Rect(
            self.width // 2 - ROAD_WIDTH // 2,
            self.height // 2 - ROAD_WIDTH // 2,
            ROAD_WIDTH,
            ROAD_WIDTH,
        )

        self.light = TrafficLight()
        self.cars: List[Car] = []

    def _spawn_car_if_needed(self, now_ms: int) -> None:
        if now_ms - self.last_spawn_ms >= SPAWN_INTERVAL_MS:
            lane = random.choice(["N", "S", "E", "W"])
            self.cars.append(Car(lane=lane))
            self.last_spawn_ms = now_ms

    def update(self) -> None:
        now_ms = pygame.time.get_ticks()
        self.light.update(now_ms)
        self._spawn_car_if_needed(now_ms)

        for car in self.cars:
            car.update(self.light.get_state(), self.intersection_rect)

        self.cars = [car for car in self.cars if not car.is_out_of_bounds(self.width, self.height)]

    def render(self) -> None:
        self.screen.fill((34, 139, 34))
        self.vertical_road.draw(self.screen)
        self.horizontal_road.draw(self.screen)

        light_color = (0, 220, 0) if self.light.get_state() == TrafficLight.GREEN else (220, 0, 0)
        pygame.draw.circle(self.screen, light_color, (self.width // 2, self.height // 2), 16)

        for car in self.cars:
            car.draw(self.screen)

        pygame.display.flip()

    def get_state(self):
        """Placeholder for future RL observation construction."""
        return {
            "light_state": self.light.get_state(),
            "num_cars": len(self.cars),
        }

    def step(self, actions=None):
        """Placeholder Gym-style step API for future RL integration."""
        self.update()
        self.render()
        reward = 0.0
        done = False
        info = {"actions_applied": actions}
        return self.get_state(), reward, done, info

    def reset(self):
        """Placeholder Gym-style reset API for future RL integration."""
        self.cars.clear()
        self.light = TrafficLight()
        self.last_spawn_ms = pygame.time.get_ticks()
        return self.get_state()

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update()
            self.render()
            self.clock.tick(FPS)

        pygame.quit()


def main() -> None:
    env = TrafficEnv()
    env.run()


if __name__ == "__main__":
    main()
