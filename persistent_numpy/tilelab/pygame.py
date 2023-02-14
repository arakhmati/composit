import pygame
from pygame import QUIT

from persistent_numpy.tilelab import Tile

COLORS = (
    (0, 255, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
)

FACTOR = 50
TAKE_OFF = 4


def visualize_tensor(surface, tilized_tensor, level=0, y=0, x=0, take_off=0):
    *_, size_y, size_x = tilized_tensor.shape

    size_y *= FACTOR
    size_x *= FACTOR

    pygame.draw.rect(
        surface,
        COLORS[level],
        pygame.Rect(x, y, size_x - take_off, size_y - take_off),
        1,
    )
    pygame.display.flip()

    if isinstance(tilized_tensor, Tile):
        return

    y += TAKE_OFF
    x += TAKE_OFF

    *_, tile_size_y, tile_size_x = tilized_tensor.tile_shape
    tile_size_y *= FACTOR
    tile_size_x *= FACTOR

    for index, tile in sorted(tilized_tensor.index_to_tile.items()):
        *_, tile_y, tile_x = index

        visualize_tensor(
            surface,
            tile,
            level + 1,
            y + tile_y * tile_size_y,
            x + tile_x * tile_size_x,
            take_off=take_off + TAKE_OFF * 2,
        )


def debug_tensors(tilized_tensor, other_tensor=None):
    pygame.init()

    surface = pygame.display.set_mode((FACTOR * 20, FACTOR * 20))
    visualize_tensor(surface, tilized_tensor)
    if other_tensor is not None:
        visualize_tensor(surface, other_tensor, x=tilized_tensor.shape[-1] * FACTOR + FACTOR)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        pygame.display.flip()

    pygame.quit()
