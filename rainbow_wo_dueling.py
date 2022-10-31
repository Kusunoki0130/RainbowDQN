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
from utils.replaybuffer_structure import NStepPREBuffer, NStepReplayBuffer
from utils.network_structure import NoisyCategoricalNetwork

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


class DQNAgent:

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
            # epsilon_decay: float,
            # max_epsilon: float = 1.0,
            # min_epsilon: float = 0.1,
            # 折扣因子
            gamma: float = 0.99,
            # PRE
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            # Categorical DQN
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-Setp Learning
            n_step: int = 3
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        # self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        # self.epsilon = max_epsilon
        # self.epsilon_decay = epsilon_decay
        # self.max_epsilon = max_epsilon
        # self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # PRE
        # 1-step
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = NStepPREBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )
        # n-step
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = NStepReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, target_dqn
        self.dqn = NoisyCategoricalNetwork(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = NoisyCategoricalNetwork(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        # NoisyNet
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
        # PRE
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss * weights)
        # samples = self.memory.sample_batch()
        # loss = self._compute_dqn_loss(samples)
        # N_step_learning
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PRE: update
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200) -> Tuple[list, list, list]:
        self.is_test = False
        state = self.env.reset()[0]
        update_cnt = 0
        # epsilons = []
        losses = []
        scores = []
        score = 0
        for frame_idx in tqdm(range(1, num_frames + 1)):
            action = self.select_action(state)
            # print(action)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

            # PRE
            franction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + franction * (1.0 - self.beta)

            if done:
                state = self.env.reset()[0]
                scores.append(score)
                score = 0
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                # self.epsilon = max(self.min_epsilon,
                #                    self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                # epsilons.append(self.epsilon)
                if update_cnt % self.target_update == 0:
                    self.dqn_target.load_state_dict(self.dqn.state_dict())
            # if frame_idx % plotting_interval == 0:
            #     self._plot(frame_idx, scores, losses, epsilons)
        self.env.close()
        self._plot(num_frames, scores, losses)
        return scores, losses

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
        self.env.close()
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
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )
        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)
        # curr_q_value = self.dqn(state).gather(1, action)
        # next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        # mask = 1 - done
        # target = (reward + self.gamma * next_q_value * mask).to(device)
        # loss = F.smooth_l1_loss(curr_q_value, target)
        return elementwise_loss

    def _plot(self, frame_idx: int, scores: List[float], losses: List[float]):
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        # plt.subplot(133)
        # plt.title('epsilons')
        # plt.plot(epsilons)
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
    memory_size = 10000
    batch_size = 128
    target_update = 100
    # epsilon_decay = 1 / 2000
    for i in range(5):
        agent = DQNAgent(env, memory_size, batch_size, target_update)
        scores, losses = agent.train(num_frames)
        with open("./save/rainbow without dueling/scores_" + str(i) + ".txt", encoding="utf-8", mode="a") as f:
            f.writelines(str(scores))
        with open("./save/rainbow without dueling/losses_" + str(i) + ".txt", encoding="utf-8", mode="a") as f:
            f.writelines(str(losses))
        # with open("./save/rainbow_dqn/epsilons_" + str(i) + ".txt", encoding="utf-8", mode="a") as f:
        #     f.writelines(str(epsilons))
        agent.save("./save/rainbow without dueling/rainbow_" + str(i) + ".pkl")
    agent = DQNAgent(env, memory_size, batch_size, target_update)
    agent.load("./save/rainbow without dueling/rainbow_4.pkl")
    video_folder = "videos/rainbow without dueling"
    agent.test(video_folder=video_folder)
