import torch
import torch.nn as nn
import torch.nn.functional as F #functions
import torch.optim as optim #optimization algorithms
import os

class Linear_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size,output_size)
        
        
    def forward(self,x):
        x = F.relu(self.Linear1(x)) #activation function
        x = self.Linear2(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr 
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done): #called for long and short, should handle different sizes # (n,x)
        #turn them to tensors
        state = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)
        
        #shaping single dimension tensors
        if len(state.shape) == 1:
            #it is (x), we need written dimension (1,x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        # 1: predict Q value with current states
        pred = self.model(state)
        
        # 2: clone pred as it gives 3 o/p values and we want one as Q_new
        # pred[argsmax(action)] = Q_new
        # Q_new = rewards + gamma * max(next_predicted Q value) -> only do this if not done
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()]  = Q_new
            
            
        self.optimizer.zero_grad() #empty the gradient?
        loss = self.criterion(target,pred)
        loss.backward() #back propagation
        self.optimizer.step()
            
            
    
     
        
        
        
        
                  
 