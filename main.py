# main.py
import sys
import pygame
from environment.game_engine import GameEngine
from environment.renderer import Renderer

import sys
import os
sys.path.append(os.path.abspath("."))  # Adds current folder to module search path


FPS = 30
WINDOW_TITLE = "Pac-Man Hybrid (Environment + UI) - Prototype"

def main():
    pygame.init()
    clock = pygame.time.Clock()
    engine = GameEngine()
    renderer = Renderer(engine)
    
    # Initial screen creation
    screen = pygame.display.set_mode((renderer.width, renderer.height))
    pygame.display.set_caption(WINDOW_TITLE)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # dt seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    engine.handle_keydown(event.key)

        # Update game
        if engine.running:
            engine.update(dt)

        renderer.render(screen)
        pygame.display.flip()

        # If game over → wait for user input
        if getattr(engine, "game_over", False):
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            engine.reset_to_level()  # Restart current level
                            
                            # ADDED: Resize window after reset
                            renderer = Renderer(engine)
                            screen = pygame.display.set_mode((renderer.width, renderer.height))
                            
                            waiting = False
                        elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            if engine.win:
                                if not engine.load_next_level():
                                    # All levels complete
                                    engine.reset_to_level(0)
                                
                                # ADDED: Resize window for next level
                                renderer = Renderer(engine)
                                screen = pygame.display.set_mode((renderer.width, renderer.height))
                                
                                waiting = False
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                
                # Keep rendering during wait
                renderer.render(screen)
                pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
