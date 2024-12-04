# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/models/ppo.py

import logging
import time
import torch as th
import numpy as np
from collections import deque
from torch.nn import functional as F

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance

from .ppo_buffer import PpoBuffer
from rl_birdview.models.discriminator import ExpertDataset


class SAC():
    # クラスの初期化SAC
    def __init__(self, policy, discriminator, env,
                 initial_learning_rate: float = 1e-5,
                 gail = True,
                 n_steps_total: int = 8192,
                 batch_size: int = 256,
                 n_epochs: int = 20,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.9,
                 clip_range: float = 0.2,
                 clip_range_vf: float = None,
                 ent_coef: float = 0.05,
                 explore_coef: float = 0.05,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 lr_decay=1.0,
                 use_exponential_lr_decay=False,
                 update_adv=False,
                 start_num_timesteps: int = 0,
                 gail_gamma=0.0,
                 gail_gamma_decay=1.0,
                 start_update: int = 0):

        self.policy = policy
        self.discriminator = discriminator
        self.env = env
        self.learning_rate = initial_learning_rate
        self.gail = gail
        self.n_steps_total = n_steps_total
        self.n_steps = n_steps_total//env.num_envs
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.explore_coef = explore_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay
        self.use_exponential_lr_decay = use_exponential_lr_decay
        self.update_adv = update_adv
        self.start_num_timesteps = start_num_timesteps
        self.num_timesteps = start_num_timesteps
        self.gail_gamma = gail_gamma
        self.gail_gamma_decay = gail_gamma_decay

        self._last_obs = None
        self._last_dones = None
        self.ep_stat_buffer = None

        # PPOのためのリプレイバッファを初期化
        #ここを変更
        self.buffer = PpoBuffer(self.n_steps, self.env.observation_space, self.env.action_space,
                                gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.env.num_envs)
        # 方策ネットワークを適切なデバイス（GPU/CPU）に移動
        self.policy = self.policy.to(self.policy.device)
        # 識別器ネットワークを適切なデバイス（GPU/CPU）に移動
        self.discriminator = self.discriminator.to(self.discriminator.device)

        # 方策ネットワークの学習可能な（勾配を計算する必要がある）パラメータのみを抽出
        model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
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
    # PpoBufferを変更
    def collect_rollouts(self, env: VecEnv, callback: BaseCallback,
                         rollout_buffer: PpoBuffer, n_rollout_steps: int) -> bool:
        # 前回の観測データが存在するか確認します
        assert self._last_obs is not None, "No previous observation was provided"
        # ステップ数を初期化
        n_steps = 0
        # ロールアウトバッファをリセットし、新しいデータ収集に備える。
        rollout_buffer.reset()

        #行動、平均値、分散、環境内のルート進行状況をを初期化
        self.action_statistics = []
        self.mu_statistics = []
        self.sigma_statistics = []
        start_route_completion = []

        while n_steps < n_rollout_steps:
            # 現在の観測データ（self._last_obs）をポリシーモデルに入力し、行動、価値などを取得
            actions, values, log_probs, mu, sigma, _, fake_birdview = self.policy.forward(self._last_obs)
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

            # ロールアウトバッファにデータを追加
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs, mu, sigma, fake_birdview, infos)
            self._last_obs = new_obs
            self._last_dones = dones

        # update_info_buffer
        for info_idx in range(len(infos)):
            route_completion = infos[info_idx]['route_completion']
            for dict_key in route_completion:
                route_completion[dict_key] -= start_route_completion[info_idx][dict_key]
            self.route_completion_buffer.append(route_completion)
        last_values = self.policy.forward_value(self._last_obs)
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
        # 収益と利得の計算
        rollout_buffer.compute_returns_and_advantage(last_values, dones=self._last_dones)

        # 完了したらTrueを返す
        return True

    def train(self):
        

    # Logs
    # 変更する可能性あり
        self.train_debug.update({
            "train/entropy_loss": np.mean(entropy_losses),
            "train/exploration_loss": np.mean(exploration_losses),
            "train/policy_gradient_loss": np.mean(pg_losses),
            "train/behavior_cloning_loss": np.mean(bc_losses),
            "train/value_loss": np.mean(value_losses),
            "train/last_epoch_kl": np.mean(approx_kl_divs),
            "train/clip_fraction": np.mean(clip_fractions),
            "train/loss": np.mean(losses),
            "train/explained_variance": explained_var,
            "train/clip_range": self.clip_range,
            "train/train_epoch": epoch,
            "train/learning_rate": self.learning_rate,
            "train/gail_gamma": self.gail_gamma
        })

    
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
    
    def learn(self, total_timesteps, callback=None, seed=2021):
        # reset env seed
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.seed(seed)

        self.start_time = time.time()

        self.t_train_values = 0.0

        self._last_obs = self.env.reset()
        self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool_)

        callback.init_callback(self)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self.ep_stat_buffer = []
            self.route_completion_buffer = []
            callback.on_rollout_start()
            t0 = time.time()
            self.policy = self.policy.train()
            continue_training = self.collect_rollouts(self.env, callback, self.buffer, self.n_steps)
            self.t_rollout = time.time() - t0
            callback.on_rollout_end()

            if continue_training is False:
                break

            t0 = time.time()
            self.train()
            if self.policy.fake_birdview and self.num_timesteps < 405504:
                self.train_gan()
            self.t_train = time.time() - t0
            callback.on_training_end()
            self.i_update += 1

        return self

    def _get_init_kwargs(self):
        init_kwargs = dict(
            initial_learning_rate=self.learning_rate,
            gail=self.gail,
            n_steps_total=self.n_steps_total,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.ent_coef,
            explore_coef=self.explore_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            lr_decay=self.lr_decay,
            use_exponential_lr_decay=self.use_exponential_lr_decay,
            update_adv=self.update_adv,
            gail_gamma=self.gail_gamma,
            gail_gamma_decay=self.gail_gamma_decay,
            start_num_timesteps=self.num_timesteps,
            start_update=self.i_update
        )
        return init_kwargs

    def save(self, path: str) -> None:
        th.save({'policy_state_dict': self.policy.state_dict(),
                 'discriminator_state_dict': self.discriminator.state_dict(),
                 'policy_init_kwargs': self.policy.get_init_kwargs(),
                 'discriminator_init_kwargs': self.discriminator.get_init_kwargs(),
                 'train_init_kwargs': self._get_init_kwargs()},
                path)

    def get_env(self):
        return self.env
