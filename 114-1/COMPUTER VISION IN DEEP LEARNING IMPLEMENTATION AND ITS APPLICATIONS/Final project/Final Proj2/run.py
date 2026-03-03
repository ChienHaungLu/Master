import os
import numpy as np
import random
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from gym.wrappers import StepAPICompatibility
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from reward import *

# 🔑 改用 gym_tetris
import gym_tetris
from reward import compute_custom_reward, SIMPLE_MOVEMENT # 從 reward.py 導入動作定義
from utils import preprocess_frame
from model import CustomCNN
from DQN import DQN, ReplayMemory
import gym

# ========== config ===========
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['A'],       # 順時針旋轉
    #['B'],       # 逆時針旋轉
    ['left'],
    ['right'],
    ['down'],    # 加速下落
]


# 1) make Tetris Environment
# TetrisA-v0: 標準模式 A
env = gym_tetris.make('TetrisA-v0')

# 2) 拆掉 TimeLimit (如果有的話)
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

# 3) 固定成舊 step API
env = StepAPICompatibility(env, output_truncation_bool=False)

# 4) 包 JoypadSpace，使用我們在 reward.py 定義的動作列表
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Final env:", env)
print("Actions:", SIMPLE_MOVEMENT)

# basic train config
LR = 0.0001 # 稍微調高一點，因為 Tetris 回饋比較稀疏
BATCH_SIZE = 32
GAMMA = 0.99
MEMORY_SIZE = 20000 # Tetris 節奏快，記憶體可以稍微大一點
EPSILON_END = 0.3  # 最終保留一點點探索
TARGET_UPDATE = 5
TOTAL_TIMESTEPS = 1000000
VISUALIZE = True   # 訓練時建議 False 加快速度
MAX_STEPS_PER_EPISODE = 5000 # 防止死循環

# 🔑 自動判斷 CPU / GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== config ===========

# DQN Initialization
obs_shape = (1, 84, 84)
n_actions = len(SIMPLE_MOVEMENT)
model = CustomCNN 
dqn = DQN(
    model=model,
    state_dim=obs_shape,
    action_dim=n_actions,
    learning_rate=LR,
    gamma=GAMMA,
    epsilon=1.0, # Start epsilon
    target_update=TARGET_UPDATE,
    device=device
)

memory = ReplayMemory(MEMORY_SIZE)
os.makedirs("ckpt_tetris", exist_ok=True) # 改個資料夾名區分
step = 0
best_reward = -float('inf') 

for timestep in tqdm(range(1, TOTAL_TIMESTEPS + 1), desc="Training Progress"):
    state = env.reset()
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0) 

    done = False
    
    # 🔑 更新 Tetris 專用的 info 追蹤
    prev_info = {
        "score": 0,
        "lines": 0
    }

    cumulative_custom_reward = 0
    cumulative_reward = 0 
    episode_steps = 0
    
    
    no_progress_steps = 0
    last_lines = 0
    last_score = 0

    while not done:
        action = dqn.take_action(state)
        

        next_state, reward, done, info = env.step(action)
        # ========== 進展偵測：避免卡局 ==========
        score = info.get("score", 0)
        

        # 只要分數或消行有增加，就算有進展
        if (score > last_score):
            no_progress_steps = 0
            last_score = score
        # ========================================

        # 防止遊戲卡死或無限進行 (Tetris 通常會自己死，但加個保險)
        episode_steps += 1
        if episode_steps > MAX_STEPS_PER_EPISODE:
            done = True

        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        # ===========================
        # 計算自定義獎勵 (注意這裡傳入 done)
        custom_reward = compute_custom_reward(info, reward, prev_info, done) 
        # ===========================

        cumulative_reward += reward # 原始 reward
        cumulative_custom_reward += custom_reward 

        # Store transition
        memory.push(state, action, custom_reward, next_state, done)
        state = next_state

        # Train DQN
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            state_dict = {
                'states': batch[0],
                'actions': batch[1],
                'rewards': batch[2],
                'next_states': batch[3],
                'dones': batch[4],
            }
            dqn.train_per_step(state_dict)

        # Update epsilon
        dqn.epsilon = max(EPSILON_END, dqn.epsilon * 0.999995) 

        prev_info = info
        step += 1

        if VISUALIZE:
            env.render()
    
    #if timestep % 10 == 0: # 減少 print 頻率
    print(f"Ep {timestep} | Total Custom Reward: {cumulative_custom_reward:.2f} | Lines: {info.get('lines', 0)}")

    if cumulative_custom_reward > best_reward:
        best_reward = cumulative_custom_reward
        model_path = os.path.join("ckpt_tetris", f"tetris_step_{timestep}_r_{int(best_reward)}.pth")
        torch.save(dqn.q_net.state_dict(), model_path)
        print(f"Model saved: {model_path}")

env.close()