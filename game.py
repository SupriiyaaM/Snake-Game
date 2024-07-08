import pygame
import random  # generating random numbers; useful for placing food in the game
from enum import Enum #: class for creating enumerations, which are a set of symbolic names bound to unique, constant values
from collections import namedtuple
import numpy as np

pygame.init() # initializes all the imported Pygame modules
font = pygame.font.Font('arial.ttf', 25)

#things we need
#reset
#reward
#play(action) -> direction
#game_iteration
#is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 40
SPEED = 40

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):  #width and height of game window
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h)) #initialize the game display with above parameters
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT  #initial direction is right
        
        self.head = Point(self.w/2, self.h/2) # initial position of the snake's head to the center of the game window
        self.snake = [self.head,                                      # snake is a list of point objects
                      Point(self.head.x-BLOCK_SIZE, self.head.y),     # second segment : one block left of head
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)] # third segment: two block left of head
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0  #reset steps to zero when reset game
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self,action): #for each time snake moves a step?
        self.frame_iteration += 1
        # 1. collect user input UNDERSTOOD
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
         
        # 3. check if game over UNDERSTOOD 
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake): #if collision happens OR if the game does not end for a long time (size of snake increases)
            game_over = True
            reward = -10         #give negative reward
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10 #reward when snake gets food   
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt= None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
         
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
  
  
        
    def _move(self, action):
        #[straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx] #no change
        elif np.array_equal(action, [0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] #right turn r -> d -> l -> u
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] #right turn r -> u -> l -> d
            
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE           
        self.head = Point(x, y)
            