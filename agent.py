import torch
import random
import numpy as np
from collections import deque #data structure to store memories
from game import SnakeGameAI, Direction, Point
from model import Linear_Qnet, QTrainer
from helper import plot

#constants
BATCH_SIZE = 1000
LR = 0.001
MAX_MEMORY = 100_000

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount factor
        self.memory = deque(maxlen = MAX_MEMORY) #pop.left() if memory exceeds
        self.model = Linear_Qnet(11,256,3) #input 11 states output 3 values
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
        #todo: model, trainer
    
    def get_state(self, game):
        head = game.snake[0] #first item of list is head
        #create 4 points around the head of the snake
        point_l = Point(head.x - 40, head.y)
        point_r = Point(head.x + 40, head.y)
        point_u = Point(head.x, head.y - 40)
        point_d = Point(head.x, head.y + 40)
        
        # any one will be true -> boolean
        dir_l  = game.direction == Direction.LEFT
        dir_r  = game.direction == Direction.RIGHT
        dir_u  = game.direction == Direction.UP
        dir_d  = game.direction == Direction.DOWN
        
        state = [
        #straight collision chances
        (dir_l and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),
        
        #right collision chances
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)) or
        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)),
        
        #left collision chances
        (dir_l and game.is_collision(point_d)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_d and game.is_collision(point_r)),
        
        #move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        
        game.food.x < game.head.x, #food left
        game.food.x > game.head.x, #food right
        game.food.y < game.head.y, #food up HOW?
        game.food.y > game.head.y, #food down
        
        ]
        
        #convert list to a numpyarray
        return np.array(state, dtype = int)
               
    
    def get_action(self, state):
        self.epsilon = 80- self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) #this will execute the forward function
            move = torch.argmax(prediction).item()
            final_move[move] = 1 #one hot encoding 
            #setting the chosen action to 1 and the rest to 0 
            
        return final_move       
    
    def remember(self, state, action, reward, next_state, done):
        #double (()) as we append as only one element
        self.memory.append((state, action, reward, next_state, done)) #pop left if memory > MAX_LENGTH
        
    def train_long_memory(self): #train from the memory
        if len(self.memory) > BATCH_SIZE: #only can take action if batch size first reaches 100 samples
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory #take whole
        
        #store states actions..etc seperately, done by zip(*)
        states, actions, rewards, next_states, dones = zip(*mini_sample)    
        self.trainer.train_step(states, actions, rewards, next_states, dones)
            
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    
def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0 # max score
    agent = Agent()
    game = SnakeGameAI()
    while True: 
        # get state
        state_old = agent.get_state(game)
        # get move to perform
        final_move = agent.get_action(state_old)
        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game) 
        #train short memory (for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        #remember all of these and store it in the memory
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done: #game is over
            #train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()            
            print('Game', agent.n_games, 'Score', score, 'Record', record)
            
            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)

if __name__ == '__main__':
    train()
    
    
    
    