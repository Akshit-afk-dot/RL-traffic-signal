import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pygame


WINDOW_SIZE = 800
FPS = 60
ROAD_WIDTH = 220
CAR_LENGTH = 28
CAR_WIDTH = 14
CAR_SPEED = 2.2
SPAWN_INTERVAL_STEPS = 12
STEP_SIMULATION_TICKS = 6
MAX_EPISODE_STEPS = 500


@dataclass
class Road:
    x: int
    y: int
    width: int
    height: int
    color: Tuple[int, int, int] = (55, 55, 55)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))


class IntersectionRenderer:
    def __init__(self, width: int, height: int, caption: str) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.closed = False

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
        return not self.closed

    def render(self, roads: List[Road], cars: List[object], light_state: int, fps: int) -> None:
        self.screen.fill((34, 139, 34))
        for road in roads:
            road.draw(self.screen)
        light_color = (0, 220, 0) if light_state == 0 else (220, 0, 0)
        pygame.draw.circle(self.screen, light_color, (self.width // 2, self.height // 2), 16)
        for car in cars:
            pygame.draw.rect(self.screen, (30, 144, 255), (*car.position, *car.size))
        pygame.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            pygame.quit()


class TrafficLight:
    def __init__(self, initial_state: int = 0) -> None:
        self.state = initial_state

    def switch(self) -> None:
        self.state = 1 - self.state

    def get_state(self) -> int:
        return self.state


@dataclass
class Car:
    lane: str
    speed: float = CAR_SPEED

    def __post_init__(self) -> None:
        lane_config: Dict[str, Dict[str, Tuple[float, float]]] = {
            "N": {"dir": (0.0, 1.0), "spawn": (WINDOW_SIZE / 2 - 25, -CAR_LENGTH)},
            "S": {"dir": (0.0, -1.0), "spawn": (WINDOW_SIZE / 2 + 11, WINDOW_SIZE + CAR_LENGTH)},
            "W": {"dir": (1.0, 0.0), "spawn": (-CAR_LENGTH, WINDOW_SIZE / 2 + 11)},
            "E": {"dir": (-1.0, 0.0), "spawn": (WINDOW_SIZE + CAR_LENGTH, WINDOW_SIZE / 2 - 25)},
        }
        if self.lane not in lane_config:
            raise ValueError(f"Unsupported lane: {self.lane}")
        self.position = list(lane_config[self.lane]["spawn"])
        self.direction = lane_config[self.lane]["dir"]
        if self.lane in {"N", "S"}:
            self.size = (CAR_WIDTH, CAR_LENGTH)
        else:
            self.size = (CAR_LENGTH, CAR_WIDTH)

    def _is_before_stop_line(self, intersection_rect: Tuple[float, float, float, float]) -> bool:
        left, top, right, bottom = intersection_rect
        x, y = self.position
        if self.lane == "N":
            return y + self.size[1] <= top
        if self.lane == "S":
            return y >= bottom
        if self.lane == "W":
            return x + self.size[0] <= left
        return x >= right

    def blocks_on_red(self, light_state: int, intersection_rect: Tuple[float, float, float, float]) -> bool:
        vertical_green = light_state == 0
        is_vertical_lane = self.lane in {"N", "S"}
        has_green = vertical_green if is_vertical_lane else not vertical_green
        return (not has_green) and self._is_before_stop_line(intersection_rect)

    def update(self, light_state: int, intersection_rect: Tuple[float, float, float, float]) -> None:
        if self.blocks_on_red(light_state, intersection_rect):
            return
        self.position[0] += self.direction[0] * self.speed
        self.position[1] += self.direction[1] * self.speed

    def is_out_of_bounds(self, width: int, height: int) -> bool:
        x, y = self.position
        return x < -100 or x > width + 100 or y < -100 or y > height + 100


class TrafficEnv:
    def __init__(self, width: int = WINDOW_SIZE, height: int = WINDOW_SIZE, render_mode: bool = False) -> None:
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.vertical_road = Road(self.width // 2 - ROAD_WIDTH // 2, 0, ROAD_WIDTH, self.height)
        self.horizontal_road = Road(0, self.height // 2 - ROAD_WIDTH // 2, self.width, ROAD_WIDTH)
        self.roads = [self.vertical_road, self.horizontal_road]
        self.intersection_rect = (
            self.width // 2 - ROAD_WIDTH // 2,
            self.height // 2 - ROAD_WIDTH // 2,
            self.width // 2 + ROAD_WIDTH // 2,
            self.height // 2 + ROAD_WIDTH // 2,
        )
        self.renderer = IntersectionRenderer(self.width, self.height, "MARL Traffic Intersection") if self.render_mode else None
        self.light = TrafficLight()
        self.cars: List[Car] = []
        self.step_count = 0
        self.state_size = 3
        self.action_size = 2

    def _spawn_car_if_needed(self) -> None:
        if self.step_count % SPAWN_INTERVAL_STEPS == 0:
            self.cars.append(Car(lane=random.choice(["N", "S", "E", "W"])))

    def _update_simulation_ticks(self, ticks: int) -> None:
        for _ in range(ticks):
            self._spawn_car_if_needed()
            for car in self.cars:
                car.update(self.light.get_state(), self.intersection_rect)
            self.cars = [car for car in self.cars if not car.is_out_of_bounds(self.width, self.height)]
            self.step_count += 1

    def get_state(self) -> np.ndarray:
        waiting_vertical = 0
        waiting_horizontal = 0
        light_state = self.light.get_state()
        for car in self.cars:
            if car.blocks_on_red(light_state, self.intersection_rect):
                if car.lane in {"N", "S"}:
                    waiting_vertical += 1
                else:
                    waiting_horizontal += 1
        return np.array([waiting_vertical, waiting_horizontal, light_state], dtype=np.float32)

    def compute_reward(self, switched: bool) -> float:
        state = self.get_state()
        waiting_penalty = -(state[0] + state[1])
        switch_penalty = -0.2 if switched else 0.0
        return float(waiting_penalty + switch_penalty)

    def step(self, action: int):
        switched = False
        if action == 1:
            self.light.switch()
            switched = True
        self._update_simulation_ticks(STEP_SIMULATION_TICKS)
        state = self.get_state()
        reward = self.compute_reward(switched)
        done = self.step_count >= MAX_EPISODE_STEPS
        info = {
            "cars_in_system": len(self.cars),
            "average_waiting_cars": float(state[0] + state[1]),
            "switched": switched,
        }
        if self.renderer is not None:
            if not self.renderer.process_events():
                done = True
            self.renderer.render(self.roads, self.cars, self.light.get_state(), FPS)
        return state, reward, done, info

    def reset(self) -> np.ndarray:
        self.light = TrafficLight(initial_state=0)
        self.cars = []
        self.step_count = 0
        return self.get_state()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
