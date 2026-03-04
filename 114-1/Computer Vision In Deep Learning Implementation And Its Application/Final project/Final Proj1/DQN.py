import torch as T
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ReplayMemory:                                                              # 存儲和取樣訓練數據
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)                                     # 使用 deque 儲存數據，設置 maxlen，確保記憶體達到容量上限時，會自動移除最舊的經驗
                                                                                 # self.memory 是一個 deque，存儲了許多 tuple，每個 tuple 表示一條經驗，格式為(s,a,r,n_s,d)
    def push(self, state, action, reward, next_state, done):                     # 將經驗 (state, action, reward, next_state, done) 添加到記憶體
        self.memory.append((state, action, reward, next_state, done))    

    def sample(self, batch_size): 
        batch = random.sample(self.memory, batch_size)                           # batch = 從記憶體中隨機取樣的 batch_size 筆資料
        states, actions, rewards, next_states, dones = zip(*batch)               # zip(*batch) 會將多筆經驗中的同類型資料（如狀態、動作）組合在一起
                                                                                 # ex.batch取樣32條(s,a,r,n_s,d)，則 zip(*batch) 中 states=[s1,s2,...,s32],actions=[a1,a2,...,a32]
        return np.stack(states), actions, rewards, np.stack(next_states), dones  # np.stack 將 states 與 next_states 轉為 NumPy 陣列，方便後續運算

    def __len__(self):                                                           # 記憶體中目前儲存的經驗數量
        return len(self.memory)                                                  

class DQN:
    def __init__(self,
                 model,
                 state_dim, action_dim, 
                 learning_rate, gamma,
                 epsilon, target_update, device):
        self.device = device
        self.action_dim = action_dim
        
        self.gamma = gamma  
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_count = 0

        # Initialize [Q-net] and target [Q-net]
        self.model = model
        self.q_net = self._build_net(state_dim, action_dim)                      #  [Q-net]，實際訓練的網路 (即時更新)
        self.tgt_q_net = self._build_net(state_dim, action_dim)                  # target [Q-net]，用於穩定訓練（延遲更新）
        self.tgt_q_net.load_state_dict(self.q_net.state_dict())                  # 複製 [Q-net] 的權重到 target [Q-net]

        # Optimizer
        self.optimizer = T.optim.Adam(self.q_net.parameters(), lr=learning_rate)
    
    # 定義神經網路構造函數
    def _build_net(self, state_dim, action_dim):                                 # 根據傳入的模型架構 (self.model) 初始化後的神經網路，並將其移到 device 上
        return self.model(state_dim,action_dim).to(self.device)
    
    # action 選擇函數
    def take_action(self, state): 
        # Exploration Unknown Policy(探索)
        if np.random.rand() < self.epsilon:                                      # 產生一個隨機浮點數（介於 0 和 1 之間），如果這個數字小於 epsilon（探索的機率），則執行隨機動作
            return np.random.randint(self.action_dim)                            # 生成一個在 [0, self.action_dim) 範圍內的隨機整數
        
        # Exploitation Known Policy(利用)                                        # 隨機浮點數大於 epsilon 執行根據推理的動作（利用）
        state_x = T.tensor([state], dtype=T.float32, device=self.device)         # 單一 state 轉換為 PyTorch 張量
        with T.no_grad():
            action_probs = F.softmax(self.q_net(state_x), dim=1)                 # 使用 [Q-net] 計算該 state 下每個動作的分數，並通過 softmax 將這些分數轉換為機率分佈
            action_dist  = T.distributions.Categorical(action_probs)             # 將機率分佈轉換為一個類別分佈物件，方便後續從中抽樣
            return action_dist.sample().item()                                   # 從類別分佈中抽樣一個動作，並返回對應的索引（即選擇的動作）
        
    #　損失函數計算
    def get_loss(self, states, actions, rewards, next_states, dones):
        # Get current Q-values
        actions = actions.unsqueeze(1) 
        q_val = self.q_net(states).gather(1, actions).squeeze(1)                 # 計算當下的 Q-value
                                                                 
        # Get maximum expected Q-values
        next_q_val = self.tgt_q_net(next_states).max(dim=1)[0]                   # 計算 target Q-value 的最大值 
                                                               
        # Compute target Q-values [custom-reward]
        q_target = rewards + self.gamma * next_q_val * (1 - dones.float())       # 計算 target Q-value
        
        return T.nn.functional.mse_loss(q_val, q_target.detach())                # 用均方誤差 (MSE) 計算 loss 

    def train_per_step(self, state_dict):
        # Convert one trajectory(s,a,r,n_s) to tensor
        states,actions,rewards,next_states,dones = self._state_2_tensor(state_dict)  # 將原本存儲於 Python 資料結構中的數據轉換為 PyTorch 張量

        # Compute loss 
        loss = self.get_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()                                               # 每次進行梯度更新之前，清除累積的梯度值
        loss.backward()                                                          # 利用計算的損失值進行反向傳播，計算每個參數的梯度
        self.optimizer.step()                                                    # 利用計算的梯度來更新 [Q-net] 的參數

        if self.update_count % self.target_update == 0:                          # runs.py 內定義 target_update=TARGET_UPDATE=50(更新頻率)
            self.tgt_q_net.load_state_dict(self.q_net.state_dict())              # 定期將 [Q-net] 的參數複製到 target [Q-net]

        self.update_count += 1
    
    def _state_2_tensor(self,state_dict):                                        # 將一條經驗軌跡 (s,a,r,n_s,d) 中的數據轉換為 PyTorch 張量
        states      = T.tensor(state_dict['states'], dtype=T.float32, device=self.device)
        actions     = T.tensor(state_dict['actions'], dtype=T.long, device=self.device)
        rewards     = T.tensor(state_dict['rewards'], dtype=T.float32, device=self.device)
        next_states = T.tensor(state_dict['next_states'], dtype=T.float32, device=self.device)
        dones       = T.tensor(state_dict['dones'], dtype=T.float32, device=self.device)

        return states,actions,rewards,next_states,dones