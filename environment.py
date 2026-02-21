from dataclasses import dataclass
from typing import Iterable, Tuple

import pygame


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

    def render(self, roads: Iterable[Road], cars: Iterable[object], light_state: int, fps: int) -> None:
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
