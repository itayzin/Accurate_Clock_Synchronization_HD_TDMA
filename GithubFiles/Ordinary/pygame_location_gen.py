import pygame
import numpy as np
import random

class InteractiveFigure:
    def __init__(self):
        self.connection_threshold = 4734.3188 / 10
        pygame.init()
        pygame.display.set_caption('Interactive Figure')
        self.screen = pygame.display.set_mode([2000, 2000])
        self.clock = pygame.time.Clock()
        self.locations = []
        self.maxLocations = 16
        self.permanent_lines = []
        self.font = pygame.font.Font(None, 24) 
        self.node_radius = 10
        self.Locations = self.run()

    def draw_dashed_line(self, start, end, color=(0,0,255), dash_length=5):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.hypot(dx, dy)
        dashes = int(length / dash_length)
        for i in range(dashes):
            if i % 2 == 0:  # Draw every other dash to create the dashed-line effect
                start_dash = (start[0] + i / dashes * dx, start[1] + i / dashes * dy)
                end_dash = (start[0] + (i+1) / dashes * dx, start[1] + (i+1) / dashes * dy)
                pygame.draw.aaline(self.screen, color, start_dash, end_dash)

    def draw_line(self, start, end, color=(0,0,255), width=1):
        pygame.draw.aaline(self.screen, color, start, end, 1)

    def draw_point(self, point, color=(0,0,255)):
        pygame.draw.circle(self.screen, (0,0,0), (int(point[0]), int(point[1])), self.node_radius)  # Draw black circumference
        pygame.draw.circle(self.screen, (255,255,255), (int(point[0]), int(point[1])), self.node_radius - 1)  # Draw white fill
        if point in self.locations:  # Only draw index if point is in locations
            text = self.font.render(str(self.locations.index(point)), True, (0, 0, 0))
            text_rect = text.get_rect(center=(point[0], point[1]))
            self.screen.blit(text, text_rect)  # Center the text

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and len(self.locations) < self.maxLocations:  # Left click
                        x, y = pygame.mouse.get_pos()
                        self.locations.append([x, y])
                        for i in range(len(self.locations) - 1):
                            if np.linalg.norm(np.array(self.locations[i]) - np.array([x, y])) <= self.connection_threshold:
                                self.permanent_lines.append(((self.locations[i], [x, y]), len(self.locations) - 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))))
                    elif event.button == 3 and len(self.locations) > 0:  # Right click
                        del self.locations[-1]
                        self.permanent_lines = [line for line in self.permanent_lines if line[1] != len(self.locations)]

            x, y = pygame.mouse.get_pos()
            self.screen.fill((255,255,255))  # Clear the screen
            for line in self.permanent_lines:
                self.draw_line(line[0][0], line[0][1], line[2], width=2)
            for i in range(len(self.locations)):
                self.draw_point(self.locations[i])
                if np.linalg.norm(np.array(self.locations[i]) - np.array([x, y])) <= self.connection_threshold:
                    self.draw_dashed_line(self.locations[i], [x, y], color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            self.draw_point([x, y], color=(0,0,255,128))
            pygame.display.flip()  # Update the screen
            self.clock.tick(120)  # Limit the frame rate to 120 FPS

        pygame.quit()
        return(np.array(self.locations))

