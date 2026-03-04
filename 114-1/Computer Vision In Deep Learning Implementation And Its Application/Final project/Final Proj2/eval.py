import numpy as np
import torch
from tqdm import tqdm
from utils import preprocess_frame
from model import CustomCNN
from DQN import DQN
import os
import gym_tetris # 🔑 Change Import
from nes_py.wrappers import JoypadSpace
from reward import SIMPLE_MOVEMENT # 🔑 Import Actions from reward.py
import gym
from gym.wrappers import StepAPICompatibility

# ========== Config ===========
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['A'],       # 順時針旋轉
    #['B'],       # 逆時針旋轉
    ['left'],
    ['right'],
    ['down'],    # 加速下落
]

# 請修改為你實際訓練出來的 Tetris 模型路徑
MODEL_PATH = os.path.join("ckpt_tetris", "tetris_step_1_r_124.pth")        

# 1) make
env = gym_tetris.make('TetrisA-v0')

# 2) Timelimit
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

# 3) Step API
env = StepAPICompatibility(env, output_truncation_bool=False)

# 4) JoypadSpace
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Final env:", env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_SHAPE = (1, 84, 84)
N_ACTIONS = len(SIMPLE_MOVEMENT) 

VISUALIZE = True
TOTAL_EPISODES = 5

# ========== Initialize DQN =========== 
dqn = DQN( 
    model=CustomCNN, 
    state_dim=OBS_SHAPE,
    action_dim=N_ACTIONS,
    learning_rate=0.0001,  
    gamma=0.99,          
    epsilon=1.0,
    target_update=5,
    device=device
)

# ========== 載入模型權重 =========== 
if os.path.exists(MODEL_PATH):
    try:
        model_weights = torch.load(MODEL_PATH, map_location=device)
        dqn.q_net.load_state_dict(model_weights)
        dqn.q_net.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        # 為了測試方便，如果沒有模型檔案，我們可以選擇報錯或用隨機權重跑跑看
        # raise
        print("Running with random weights for testing...")
else:
    print(f"Model file not found: {MODEL_PATH}, using random weights.")

# ========== Evaluation Loop ===========
for episode in range(1, TOTAL_EPISODES + 1):
    state = env.reset()
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)
    state = np.expand_dims(state, axis=0)
    
    done = False
    total_reward = 0
    lines_cleared = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_probs = torch.softmax(dqn.q_net(state_tensor), dim=1)
            action = torch.argmax(action_probs, dim=1).item()
            
        next_state, reward, done, info = env.step(action)
        
        # 記錄行數
        lines_cleared = info.get('lines', 0)

        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        total_reward += reward
        state = next_state

        if VISUALIZE:
            env.render()

    print(f"\nEpisode {episode}/{TOTAL_EPISODES} - Lines Cleared: {lines_cleared}")

env.close()