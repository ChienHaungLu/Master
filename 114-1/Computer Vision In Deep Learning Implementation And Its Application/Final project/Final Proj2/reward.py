import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import time

# 記錄目前方塊的生成時間
current_piece_start_time = None

# Tetris Actions (自定義適合 Tetris 的動作)
# 對應 NES 手把: ['B', 'NULL', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
# 這裡我們定義簡化動作
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['A'],       # 順時針旋轉
    #['B'],       # 逆時針旋轉
    ['left'],
    ['right'],
    ['down'],    # 加速下落
]

# Env state info keys usually in gym-tetris:
# "score": (int)
# "lines": (int) number of lines cleared
# "board_height": (int) max height of bricks (sometimes available depending on wrapper)

#-----------------------------------------------------------------------------
# 獎勵函數定義
#-----------------------------------------------------------------------------

# 1. 分數獎勵：當有分數時給予獎勵
def get_score_reward(info, reward, prev_info):
    total_reward = reward
    # 計算分數差值
    delta_score = info.get('score', 0) - prev_info.get('score', 0)
    
    if delta_score >= 0:
        # 分數增加通常代表消除或下落，給予適當權重
        total_reward += delta_score * 0.1 + 0.01
    
    return total_reward

# 2. 消除行數獎勵：這是 Tetris 的核心目標
def get_lines_reward(info, reward, prev_info):
    total_reward = reward
    # gym-tetris 的 info 中通常包含 'lines' (已消除總行數)
    # 注意：有些版本可能是 'number_of_lines'
    current_lines = info.get('lines', 0)
    prev_lines = prev_info.get('lines', 0)
    
    delta_lines = current_lines - prev_lines
    
    if delta_lines > 0:
        # 消除行數給予巨大獎勵 (鼓勵一次消多行)
        
        total_reward += delta_lines * 2
        
    return total_reward

# 3. 存活/懲罰機制
def survival_reward(info, reward, done):
    total_reward = reward
    
    # 每存活一步給予微小獎勵，鼓勵不要立刻死掉 (但要小心不要造成 Agent 故意不消行只為了拖時間)
    # 在 Tetris 中，通常不建議給每一步太高的存活獎勵，因為會導致 Agent 堆疊而不消除
    total_reward += random.uniform(0.0, 0.05) #隨機給予獎勵

    # 死亡懲罰
    if done:
        total_reward -= 50.0
        
    return total_reward

# 4. 下落獎勵
def too_long(info,reward):
    """
    若單一方塊騰空時間 > 5 秒，扣 3 分
    否則（成功快速落地）加 1 分
    """

    global current_piece_start_time
    total_reward = reward

    # 如果是新方塊生成
    if current_piece_start_time is None:
        current_piece_start_time = time.time()

    # 判斷方塊是否剛剛落地 / 鎖定
    # key 名稱可能依 wrapper 不同
    piece_locked = info.get("piece_locked", False)

    if piece_locked:
        air_time = time.time() - current_piece_start_time

        if air_time > 5.0:
            total_reward -= 3.0
        else:
            total_reward += 1.0

        # 重置，準備下一個方塊
        current_piece_start_time = None

    return total_reward


# function彙整
def compute_custom_reward(info, reward, prev_info, done=False):
    # gym-tetris 原始 reward 通常只是 0 或 1 (代表是否活著)，我們主要依賴自定義
    r = 0
    r = get_score_reward(info, r, prev_info)
    r = get_lines_reward(info, r, prev_info)
    r = survival_reward(info, r, done)
    r = too_long(info,r)
    return r