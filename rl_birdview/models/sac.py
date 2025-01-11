# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/models/ppo.py

import logging
import time
import torch as th
import numpy as np
from collections import deque
from torch.nn import functional as F
from copy import deepcopy

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from .ppo_buffer import ReplayBuffer
from rl_birdview.models.discriminator import ExpertDataset

# SAcクラス
class SAC():
    def __init__(self, policy, discriminator, env, 
                 initial_actor_lr = 1e-5, 
                 initial_critic_lr = 1e-5,
                 gail = True,
                 n_steps_total: int = 8192,
                 batch_size: int = 256,
                 n_epochs: int = 20,
                 gamma: float = 0.99,             # バッファに使用
                 gae_lambda: float = 0.9,         # バッファに使用
                 buffer_size = 1_000_000, 
                 tau=0.005,                       # targetNETの更新に使用
                 alpha=0.2,                       # Q値の計算（SAC特有）
                 ent_coef: float = 0.05,          # lossの計算
                 explore_coef: float = 0.0,       # lossの計算
                 vf_coef: float = 0.5,            # lossの計算
                 max_grad_norm: float = 0.5,      # 勾配爆発を抑えるのに使用
                 lr_decay=1.0,                    # 学習率で使用
                 use_exponential_lr_decay=False,  # 学習率で使用
                 start_num_timesteps: int = 0,
                 gail_gamma=0.0,                  # GAILの学習率
                 gail_gamma_decay=1.0,            # GAILの学習率減衰
                 start_update: int = 0):

        self.discriminator = discriminator
        self.env = env
        self.actor_lr = initial_actor_lr
        self.critic_lr = initial_critic_lr
        self.gail = gail
        self.n_steps_total = n_steps_total
        self.n_steps = n_steps_total//env.num_envs
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma 
        self.gae_lambda = gae_lambda 
        self.buffer_size = buffer_size
        self.tau = tau 
        self.alpha = alpha 
        self.ent_coef = ent_coef 
        self.explore_coef = explore_coef 
        self.vf_coef = vf_coef 
        self.max_grad_norm = max_grad_norm 
        self.lr_decay = lr_decay 
        self.use_exponential_lr_decay = use_exponential_lr_decay 
        self.start_num_timesteps = start_num_timesteps
        self.num_timesteps = start_num_timesteps
        self.gail_gamma = gail_gamma 
        self.gail_gamma_decay = gail_gamma_decay  

        self._last_obs = None
        self._last_dones = None
        self.ep_stat_buffer = None
        
 #       def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space,
 #                warmup_steps: int = 2e3, gamma: float = 0.99, tau: float = 0.005, alpha = 0.001, n_envs: int = 1):

        # SACのためのリプレイバッファを初期化
        self.buffer = ReplayBuffer(self.buffer_size, self.n_steps_total, self.env.observation_space, self.env.action_space,
                                gamma=self.gamma, n_envs=self.env.num_envs)
    
        # 方策ネットワークを適切なデバイス（GPU/CPU）に移動
        self.policy = policy
        self.policy = self.policy.to(self.policy.device)
        # ターゲットの方策ネットワークを適切なデバイス（GPU/CPU）に移動
        self.target_policy = deepcopy(self.policy)
        self.target_policy = self.target_policy.to(self.policy.device)

        # 識別器ネットワークを適切なデバイス（GPU/CPU）に移動
        self.discriminator = self.discriminator.to(self.discriminator.device)

        # 方策ネットワークの学習可能な（勾配を計算する必要がある）パラメータのみを抽出
        model_parameters = filter(lambda p: p.requires_grad, self.policy.actor_model.parameters())
        # 方策ネットワーク内の総パラメータ数を計算
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        # 判別器の学習可能なパラメータを抽出
        disc_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        # 方策・判別器の総パラメータ数を計算し、出力
        total_params += sum([np.prod(p.size()) for p in disc_parameters])
        print(f'trainable parameters: {total_params/1000000:.2f}M')
        
        self.logger = logging.getLogger(__name__)
        self.train_debug = {}
        self.i_update = start_update

    # 環境の行動結果を記録して、強化学習に必要なデータをまとめる
    def collect_rollouts(self, env: VecEnv, callback: BaseCallback,
                         rollout_buffer: ReplayBuffer, n_rollout_steps: int) -> bool:
        # 前回の観測データが存在するか確認します
        assert self._last_obs is not None, "No previous observation was provided"
        # ステップ数を初期化
        n_steps = 0

        #行動、平均値、分散、環境内のルート進行状況をを初期化
        self.action_statistics = []
        self.mu_statistics = []
        self.sigma_statistics = []
        start_route_completion = []
        
        while n_steps < n_rollout_steps:
            # 現在の観測データ（self._last_obs）をポリシーモデルに入力し、行動、価値などを取得
            actions, log_probs, mu, sigma, _, fake_birdview = self.policy.forward(self._last_obs)
            self.action_statistics.append(actions)
            self.mu_statistics.append(mu)
            self.sigma_statistics.append(sigma)

            # 環境にアクションを入力し、次の状態（new_obs）、報酬（rewards）、エピソード終了フラグ（dones）、追加情報（infos）を取得
            new_obs, rewards, dones, infos = env.step(actions)

            # コールバック関数を呼び出し、ステップごとの処理を実行。
            if callback.on_step() is False:
                return False

            # ０ステップ時の処理
            if n_steps == 0:
                for info in infos:
                    start_route_completion.append(info['route_completion'])

            # ステップ数を増加し、更新する
            n_steps += 1
            self.num_timesteps += env.num_envs

            # update_info_buffer
            # エピソード終了時の処理
            for i in np.where(dones)[0]:
                self.ep_stat_buffer.append(infos[i]['episode_stat'])
                if n_steps < n_rollout_steps:
                    route_completion = infos[i]['route_completion']
                    for dict_key in route_completion:
                        route_completion[dict_key] -= start_route_completion[i][dict_key]
                    self.route_completion_buffer.append(route_completion)

            for i in np.where(self._last_dones)[0]:
                start_route_completion[i] = infos[i]['route_completion']

            # リプレイバッファにデータを追加(現在の状態、行動、報酬、次の状態、終了条件、)
            rollout_buffer.add(self._last_obs, actions, rewards, new_obs, dones, fake_birdview, infos)
            self._last_obs = new_obs
            self._last_dones = dones

        # update_info_buffer
        for info_idx in range(len(infos)):
            route_completion = infos[info_idx]['route_completion']
            for dict_key in route_completion:
                route_completion[dict_key] -= start_route_completion[info_idx][dict_key]
            self.route_completion_buffer.append(route_completion)

        # 識別器の更新
        if self.gail:
            self.discriminator.update(self.buffer)
            # バッファ内のデータに基づいて報酬を予測
            for step in range(rollout_buffer.buffer_size):
                obs_dict = dict([(obs_key, th.as_tensor(obs_item[step])) for obs_key, obs_item in rollout_buffer.observations.items()])
                rollout_buffer.rewards[step] = self.discriminator.predict_reward(
                    obs_dict,
                    th.as_tensor(rollout_buffer.actions[step]),
                    rollout_buffer.gamma,
                    th.as_tensor(rollout_buffer.dones[step]),
                ).detach().numpy().reshape(-1)

        # 完了したらTrueを返す
        return True


    # Q関数の学習
    def train(self):
        # Actorの学習率を更新
        for param_group in self.policy.actor_model.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_lr
        # Criticの学習率を更新
        for param_group in self.policy.critic_model.critic_optimizer.param_groups:
            param_group["lr"] = self.critic_lr

        # 初期化
        entropy_losses, exploration_losses, actor_losses, bc_losses, critic_losses, losses = [], [], [], [], [], []
        # train for gradient_steps epochs
        epoch = 0
        # 更新回数を計算
        data_len = int(self.buffer.buffer_size * self.buffer.n_envs / self.batch_size)

        # fake_birdview が有効化されている場合、GANによる疑似バードビュー生成を行う
        if self.policy.fake_birdview:
            data_loader_fake_birdview = self.policy.gan_fake_birdview.fill_expert_dataset(self.discriminator.expert_loader)

        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            # バッファからのデータを取得
            self.buffer.start_caching(self.batch_size)
            # while self.buffer.sample_queue.qsize() < 3:
                # time.sleep(0.01)
            for i in range(data_len):
                rollout_data = self.buffer.sample_queue.get()

                # エントロピー損失,探索損失を計算
                entropy_loss, exploration_loss = self.policy.evaluate_actions(rollout_data.observations, rollout_data.actions, rollout_data.exploration_suggests, rollout_data.fake_birdviews)
                
                # criticの損失
                critic_loss = self._critic_learn(rollout_data.observations, rollout_data.actions, rollout_data.rewards, rollout_data.next_observations, rollout_data.dones)
               # actorの損失
                actor_loss = self._actor_learn(rollout_data.observations) 

                # Expert dataset
                # 専門家データからBC損失（専門家の損失）を計算する
                for expert_batch in self.discriminator.expert_loader:
                    expert_obs_dict, expert_action = expert_batch
                    obs_tensor_dict = dict([(obs_key, obs_item.float().to(self.policy.device)) for obs_key, obs_item in expert_obs_dict.items()])
                    fake_birdview = None
                    if self.policy.fake_birdview:
                        fake_birdview = data_loader_fake_birdview.index_select(dim=0, index=expert_obs_dict['item_idx'].int())
                        fake_birdview = fake_birdview.to(self.policy.device)
                    expert_action = expert_action.to(self.policy.device)
                    # Get BC loss
                    alogprobs, _ = self.policy.evaluate_actions_bc(obs_tensor_dict, fake_birdview, expert_action)
                    bcloss = -alogprobs.mean()
                    # action loss is weighted sum
                    action_loss = self.gail_gamma * bcloss + (1 - self.gail_gamma) * actor_loss
                    # Multiply this coeff with decay factor
                    break

                # 損失の合計
                loss = action_loss + self.vf_coef * critic_loss \
                    + self.ent_coef * entropy_loss + self.explore_coef * exploration_loss

                # 計算された損失をログに記録
                losses.append(loss.item())
                actor_losses.append(actor_loss.item())
                bc_losses.append(bcloss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                exploration_losses.append(exploration_loss.item())

                # Optimization step
                # 勾配を計算する前に以前の勾配をリセット
                self.policy.actor_model.actor_optimizer.zero_grad()
                self.policy.critic_model.critic_optimizer.zero_grad()

                # loss（全体の損失）に基づいて、各パラメータの勾配が計算
                loss.backward()

                # 勾配爆発（gradient explosion）を防ぐために、勾配のノルム（大きさ）を制限
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                # パラメータの更新
                self.policy.actor_model.actor_optimizer.step()
                self.policy.critic_model.critic_optimizer.step()

                 # targetネットワークの更新
                self.sync_target()

        if self.use_exponential_lr_decay:
            self.actor_lr *= self.lr_decay
            self.critic_lr *= self.lr_decay

        if self.gail_gamma is not None:
            self.gail_gamma *= self.gail_gamma_decay

        # ログの更新
        self.train_debug.update({
            "train/entropy_loss": np.mean(entropy_losses),
            "train/exploration_loss": np.mean(exploration_losses),
            "train/actor_loss": np.mean(actor_losses),
            "train/behavior_cloning_loss": np.mean(bc_losses),
            "train/critic_loss": np.mean(critic_losses),
            "train/loss": np.mean(losses),
            "train/train_epoch": epoch,
            "train/actor_lr": self.actor_lr,
            "train/critic_lr": self.critic_lr,
            "train/gail_gamma": self.gail_gamma
        })


    # Critic(評価器)の計算
    def _critic_learn(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            # 次の行動をサンプリング、対数確率を計算
            next_action, next_log_pro = self.sample(next_obs)
            # 次の状態のQ値を計算
            q1_next, q2_next = self.target_policy.critic_model(next_obs, next_action)
            # ターゲット値を計算
            target_Q = th.min(q1_next, q2_next) - self.alpha * next_log_pro
            target_Q = reward + self.gamma * (1. - done) * target_Q
            # 現在のQ値を計算
            q1_cur, q2_cur = self.policy.critic_model(obs, action)
        # １ステップ勾配降下法で損失を計算
        critic_loss = F.mse_loss(q1_cur, target_Q) + F.mse_loss(q2_cur, target_Q)

        # critic損失を返す
        return critic_loss


    # Actor(行動器)の学習
    def _actor_learn(self, obs):
        # 現在の行動をサンプリングし、その対数確率を計算
        action, log_probs = self.policy.actor_model(obs)
        # 現在の状態のQ値を計算
        q1_pi, q2_pi = self.policy.critic_model(obs, action)
        # １ステップ勾配法で損失を計算
        min_q_pi = th.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_probs) - min_q_pi).mean()

        # actor損失を返す
        return actor_loss


    # Targetネットワークの更新
    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.policy.sync_weights_to(self.target_policy, decay=decay)


    # GANの事前学習（変更なし）
    def train_gan(self):
        gan_disc_losses, gan_generator_losses, gan_pixel_losses, gan_losses = [], [], [], []
        gan_epochs = 4
        gan_batch_size = 32
        for _ in range(gan_epochs):
            for policy_batch in self.buffer.get(gan_batch_size):
                gan_disc_loss, gan_generator_loss, gan_pixel_loss, gan_loss = self.policy.gan_fake_birdview.train_batch(policy_batch.observations, self.num_timesteps)
                gan_disc_losses.append(gan_disc_loss)
                gan_generator_losses.append(gan_generator_loss)
                gan_pixel_losses.append(gan_pixel_loss)
                gan_losses.append(gan_loss)

        self.train_debug.update({
            "train_gan/gan_disc_loss": np.mean(gan_disc_losses),
            "train_gan/gan_generator_loss": np.mean(gan_generator_losses),
            "train_gan/gan_pixel_loss": np.mean(gan_pixel_losses),
            "train_gan/gan_loss": np.mean(gan_losses)
        })
    
    #学習を進める関数
    def learn(self, total_timesteps, callback=None, seed=2021):
        # reset env seed
        # 環境の乱数シード値を設定
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.seed(seed)

        # ログの記録用に学習開始時刻を取得
        self.start_time = time.time()

        # 意味のない変数
        self.t_train_values = 0.0

        # 各環境をリセットし、初期の状態を取得
        self._last_obs = self.env.reset()
        # 各環境の終了条件を全てFalseに初期化
        self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool_)

        #　コールバックの初期化
        callback.init_callback(self)

        #　コールバックを使用して学習開始の処理をする（ログの記録や初期設定）
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self.ep_stat_buffer = []
            self.route_completion_buffer = []
            callback.on_rollout_start()
            # 現在の時間を取得
            t0 = time.time()
            # 方策学習
            self.policy = self.policy.train()
            # 環境からデータを収集
            continue_training = self.collect_rollouts(self.env, callback, self.buffer, self.n_steps)
            # データ収集にかかった時間を計測
            self.t_rollout = time.time() - t0
            callback.on_rollout_end()

            if continue_training is False:
                break

            # 現在の時間を取得
            t0 = time.time()
            # 学習を行う
            self.train()
            # GANの学習を行う
            if self.policy.fake_birdview and self.num_timesteps < 405504:
                self.train_gan()
            # 学習時間を計測
            self.t_train = time.time() - t0
            callback.on_training_end()
            # 更新回数を更新
            self.i_update += 1

        return self

    def _get_init_kwargs(self):
        init_kwargs = dict(
            initial_actor_lr=self.actor_lr,
            initial_critic_lr=self.critic_lr,
            gail=self.gail,
            n_steps_total=self.n_steps_total,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            buffer_size=self.buffer_size,
            tau=self.tau,
            alpha=self.alpha,
            ent_coef=self.ent_coef,
            explore_coef=self.explore_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            lr_decay=self.lr_decay,
            use_exponential_lr_decay=self.use_exponential_lr_decay,
            start_num_timesteps=self.num_timesteps,
            gail_gamma=self.gail_gamma,
            gail_gamma_decay=self.gail_gamma_decay,
            start_update=self.i_update
        )
        return init_kwargs


    # モデルの保存(変更なし)
    def save(self, path: str) -> None:
        th.save({'policy_state_dict': self.policy.state_dict(),
                 'discriminator_state_dict': self.discriminator.state_dict(),
                 'policy_init_kwargs': self.policy.get_init_kwargs(),
                 'discriminator_init_kwargs': self.discriminator.get_init_kwargs(),
                 'train_init_kwargs': self._get_init_kwargs()},
                path)

    # 環境を取得する関数（変更なし）
    def get_env(self):
        return self.env
