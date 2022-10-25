import os
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.replaybuffer_structure import NStepReplayBuffer
from utils.network_structure import Network

"""
requirements:
torch              
matplotlib          
pyglet              
gym                 
PyVirtualDisplay
moviepy
pygame 
"""

"""
class DQNAgent:

Method:
    select_action : 从动作空间中使用 ε-greedy 策略选择 action
    step : 根据所选 action，计算下一个 state
    compute_dqn_loss : 计算损失
    update_model : 更新模型
    train : 训练
    test : 测试
    plot : 展示
"""


class NStepDQNAgent:

    def __init__(
            self,
            # gym 提供的游戏环境
            env: gym.Env,
            # 样本空间大小
            memory_size: int,
            # 批训练样本数量
            batch_size: int,
            # 模型参数硬更新周期
            target_update: int,
            # ε-greedy
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            # 折扣因子
            gamma: float = 0.99,
            # N_step_learning
            n_step: int = 3
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env

        # N_step_learning
        self.memory = NStepReplayBuffer(obs_dim, memory_size, batch_size, n_step=1)
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = NStepReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)

        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks: dqn, target_dqn
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        # 从动作空间中使用 ε-greedy 策略选择 action
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _1, _2 = self.env.step(action)
        if not self.is_test:
            self.transition += [reward, next_state, done]
            # N_step_learning
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            else:
                one_step_transition = self.transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
        return next_state, reward, done

    def update_model(self) -> float:
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        loss = self._compute_dqn_loss(samples, self.gamma)

        # N_step_learning
        if self.use_n_step:
            samples = self.memory_n.sample_batch_from_idxs(indices)
            gamma = self.gamma ** self.n_step
            n_loss = self._compute_dqn_loss(samples, gamma)
            loss += n_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200) -> Tuple[list, list, list]:
        self.is_test = False
        state = self.env.reset()[0]
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        for frame_idx in tqdm(range(1, num_frames + 1)):
            action = self.select_action(state)
            # print(action)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
            if done:
                state = self.env.reset()[0]
                scores.append(score)
                score = 0
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                self.epsilon = max(
                    self.min_epsilon,
                    self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                if update_cnt % self.target_update == 0:
                    self.dqn_target.load_state_dict(self.dqn.state_dict())
            # if frame_idx % plotting_interval == 0:
            #     self._plot(frame_idx, scores, losses, epsilons)
        self.env.close()
        # self._plot(num_frames, scores, losses, epsilons)
        return scores, losses, epsilons

    def test(self, video_folder: str) -> None:
        self.is_test = True
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        state = self.env.reset()[0]
        done = False
        score = 0
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

        print("score: ", score)
        self.env = naive_env

    def save(self, path):
        torch.save(self.dqn.state_dict(), path)

    def load(self, path):
        self.dqn.load_state_dict(torch.load(path))
        self.epsilon = 0.1

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(device)
        loss = F.smooth_l1_loss(curr_q_value, target)
        return loss

    def _plot(self, frame_idx: int, scores: List[float], losses: List[float], epsilons: List[float]):
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    env_id = "CartPole-v1"
    env = gym.make(env_id, render_mode="rgb_array")
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)
    # env.seed(seed)

    num_frames = 20000
    memory_size = 2000
    batch_size = 32
    target_update = 100
    epsilon_decay = 1 / 2000
    # for i in range(5):
    #     agent = NStepDQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
    #     scores, losses, epsilons = agent.train(num_frames)
    #     with open("./save/n_step_learning_dqn/scores_" + str(i) + ".txt", encoding="utf-8", mode="a") as f:
    #         f.writelines(str(scores))
    #     with open("./save/n_step_learning_dqn/losses_" + str(i) + ".txt", encoding="utf-8", mode="a") as f:
    #         f.writelines(str(losses))
    #     with open("./save/n_step_learning_dqn/epsilons_" + str(i) + ".txt", encoding="utf-8", mode="a") as f:
    #         f.writelines(str(epsilons))
    #     agent.save("./save/n_step_learning_dqn/n_step_learning_dqn_" + str(i) + ".pkl")
    agent = NStepDQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
    agent.load("./save/n_step_learning_dqn/n_step_learning_dqn_3.pkl")
    video_folder = "videos/n_step_learning_dqn"
    agent.test(video_folder=video_folder)
