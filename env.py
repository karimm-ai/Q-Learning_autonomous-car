from gym import Env, spaces
import pygame
import numpy as np


class proj(Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
    
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.action_space = [0,1,2,3] # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.steps = 0
        
        self._action_to_direction = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }
        
        self.observation_space = spaces.Box(0, size -1, shape=(2,), dtype=int)
        self._agent_location = (4,0)
        self._target_location = (0,4)
        
        #Our 3 obstacles
        self._obstacle_location = [(0,3), (3,1), (4,1)]
        
        #Our two pedestrians
        self._pedestrian_location = [(0,0), (2,3)] 
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode        
        self.window = None
        self.clock = None

    
    def hit_obstacle(self):
        return self._agent_location in self._obstacle_location
    
    
    def hit_pedestrian(self):
        return self._agent_location in self._pedestrian_location
    
        
    def _get_observation(self):
        return self._agent_location
    
        
    def delivered_pizza(self):
        return self._agent_location == self._target_location
    
    
    def off_grid(self):
        x, y = self._agent_location
        return x < 0 or x >= self.size or y < 0 or y >= self.size
    
    
    def step(self, action):
        self.steps += 1
        reward = -1
        done = False
        direction = self._action_to_direction[action]
        self._agent_location = tuple(np.array(self._agent_location) + np.array(direction))
        
        if self.hit_obstacle() or self.off_grid():
            reward -= 5
            #The agent returns back to its last position after hitting an obstacle or going off-grid.
            self._agent_location = tuple(np.array(self._agent_location) - np.array(direction))
 
            
        if self.hit_pedestrian():
            reward -= 20
            
            
        if self.delivered_pizza():
            reward += 10
            done = True
                
        observation = self._agent_location
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done 

    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * np.array(self._target_location),
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (100, 100, 100),
            (np.array(self._agent_location) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
        #Drawing the Obstacles
        for obs in self._obstacle_location:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * np.array(obs),
                    (pix_square_size, pix_square_size),
                ),
            )
            
        #Drawing the pedestrians
        for ped in self._pedestrian_location:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * np.array(ped),
                    (pix_square_size, pix_square_size),
                ),
            )            


        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

            
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    
    def reset(self):
        #Reset the agent's and pedestrians' positions
        self._agent_location = (4,0)
        self._pedestrian_location = [(0,0), (2,3)]

        observation = self._agent_location
        self.steps = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation