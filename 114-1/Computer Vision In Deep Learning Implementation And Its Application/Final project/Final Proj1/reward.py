import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Env state 
info = {
     "x_pos",  # (int) The player's horizontal position in the level.
     "y_pos",  # (int) The player's vertical position in the level.
     "score",  # (int) The current score accumulated by the player.
     "coins",  # (int) The number of coins the player has collected.
     "time",   # (int) The remaining time for the level.
     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
     "life"   # (int) The number of lives the player has left.
}


 # simple actions_dim = 7 
SIMPLE_MOVEMENT = [
    ["right"],
    ["right","B"],
    ["right","A"],
    ["right","A","B"],
    ["A"],
    ["left"],          # 只留最基本左移
]
#-----------------------------------------------------------------------------
#獎勵函數
'''
get_coin_reward         : 根據硬幣數量變化提供額外獎勵

'''
'''
環境資訊 (info)
1."x_pos": 水平位置，用於判斷角色的前進情況
2."y_pos": 垂直位置，用於分析跳躍或下落行為
3."score": 玩家目前的遊戲分數
4."coins": 收集到的硬幣數量
5."time": 剩餘時間
5."flag_get": 是否到達終點旗幟（遊戲完成）
6."life": 玩家剩餘的生命數
'''

#===============to do===============================請自定義獎勵函數 至少7個(包含提供的)
#例子:用來獎勵玩家蒐集硬幣的行為
def get_coin_reward(info, reward, prev_info):
    #寫下蒐集到硬幣會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量

    total_reward += (info['coins'] - prev_info['coins'])*10     #這裡是定義，如果玩家有蒐集到硬幣，則獎勵乘10(這裡是可以自己去定義獎勵要給多少的)
    return total_reward

#用來鼓勵玩家進行跳躍或高度變化(因為有時前方有障礙物 會被卡住)
def distance_y_offset_reward(info, reward, prev_info):
    """
    Encourage jump / vertical movement to avoid getting stuck by obstacles.
    Note: In gym-super-mario-bros, y_pos usually increases when moving DOWN.
          Jumping UP typically makes y_pos decrease.
    """
    total_reward = reward

    dy = info["y_pos"] - prev_info["y_pos"]

    # Reward upward movement (jump): dy < 0 means y decreased
    if dy < 0:
        # small positive reward for jumping up
        total_reward += min((-dy) * 0.02, 1.0)

    # Penalize big downward drops (falling into pits / off platforms)
    if dy > 0:
        # Small bonus for any vertical change (encourage exploration)
        total_reward -= min(abs(dy) * 0.01, 0.5)

    return total_reward

#用來鼓勵玩家前進，懲罰原地停留或後退
def distance_x_offset_reward(info, reward, prev_info):
    """
    Encourage moving right (progress), punish staying still / moving backward.
    """
    total_reward = reward
    dx = info["x_pos"] - prev_info["x_pos"]

    if dx > 0:
        total_reward += min(dx * 0.05, 2.0)
    elif dx == 0:
        total_reward -= 0.1
    else:
        total_reward -= min((-dx) * 0.05, 1.0)

    return total_reward

#用來鼓勵玩家提高分數（例如擊敗敵人)
def monster_score_reward(info, reward, prev_info):
    """
    Reward score increase (often from defeating enemies / hitting blocks).
    Also penalize losing a life.
    """
    total_reward = reward

    dscore = info["score"] - prev_info["score"]
    if dscore > 0:
        # scale down score reward to avoid exploding reward
        total_reward += min(dscore * 0.05, 10.0)

    # Penalize life loss heavily (dying is bad)
    if "life" in info and "life" in prev_info:
        if info["life"] < prev_info["life"]:
            total_reward -= 50.0

    return total_reward

#用來鼓勵玩家完成關卡（到達終點旗幟）
def final_flag_reward(info,reward):
    """
    Big reward for completing the level.
    """
    total_reward = reward
    if info.get("flag_get", False):
        total_reward += 500.0
    return total_reward


#動機: 有限時間內完成取得特定得分(在剩餘320秒內取得1000分，即可直接通關)
#預期: 透過此函數設計讓agent盡可能在有限時間內達標
def time_constraint_score(info,reward):
    """
    Motivation:
      Within remaining 320 seconds, reach score >= 10000 => give a big bonus.
    Expected:
      Encourage the agent to play efficiently (faster high-score route).
    """
    total_reward = reward
    if info.get("time", 0) >= 320 and info.get("score", 0) >= 1000:
        total_reward += 150.0
    return total_reward



#動機: 在剩370秒內至少取得250分，否則每隔3秒扣10分，直到0分就視為失敗
#預期: 讓agent在有限時間內取得一定分數，來避免遊戲提前結束
def time_constraint_score2(info, reward):
    """
    Motivation:
      Within remaining 250 seconds, the agent should reach score >= 300.
      Otherwise:
        - Every 3 seconds: apply a -10 penalty
        - If score drops to 0 or below near deadline: strong failure penalty

    Expected:
      Encourage early score acquisition and avoid passive play.
    """
    total_reward = reward
    t = info.get("time", 0)
    s = info.get("score", 0)

    if t <= 370 and s < 300:
        if t % 3 == 0:
            total_reward -= 10.0
        if s <= 0 and t <= 50:
            total_reward -= 100.0
    return total_reward

#動機: 至少存活超過7秒，否則直接扣200分，反之存活超過7秒可得50分
#預期: 讓agent盡可能存活超過7秒
def alive(info,reward):
    """
    Motivation:
      Survive at least 7 seconds.
    Rule:
      - If agent dies before 7 seconds: -500
      - If agent survives past 7 seconds: +50 (one-time bonus)
    """
    total_reward = reward
    initial_time = 400
    survived_time = initial_time - info.get("time", initial_time)

    if survived_time == 7:
        total_reward += 50.0
    if survived_time < 7 and info.get("life", 1) <= 0:
        total_reward -= 200.0

    return total_reward

#===============to do==========================================

#function彙整
def compute_custom_reward(info, reward, prev_info):
    r = reward
    r = get_coin_reward(info, r, prev_info)
    r = distance_x_offset_reward(info, r, prev_info)
    r = distance_y_offset_reward(info, r, prev_info)
    r = monster_score_reward(info, r, prev_info)
    r = final_flag_reward(info, r)
    r = time_constraint_score(info, r)
    r = time_constraint_score2(info, r)
    r = alive(info,r)
    return r