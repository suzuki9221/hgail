# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/models/ppo_policy.py

from typing import Union, Dict, Tuple, Any
from functools import partial
import gym
import torch as th
import torch.nn as nn
import numpy as np
from PIL import Image

from carla_gym.utils.config_utils import load_entry_point

from rl_birdview.models.gan_layers import GanFakeBirdview

# Actorクラス
class Actor(nn.Module):
    def __init__(self, action_space, features_extractor, policy_head_arch, action_dist, device='cpu'):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.policy_head_arch = policy_head_arch
        self.action_dist = action_dist
        self.device = device
        self.activation_fn = nn.ReLU

        self.actor = self._build()

    # Actorネットワークの構築
    def _build(self):
        last_layer_dim = self.features_extractor.features_dim
        actor_net = []
        for layer_size in self.policy_head_arch:
            actor_net.append(nn.Linear(last_layer_dim, layer_size))
            actor_net.append(self.activation_fn())
            last_layer_dim = layer_size

        self.dist_mu, self.dist_sigma = self.action_dist.proba_distribution_net(last_layer_dim)
        return nn.Sequential(*actor_net).to(self.device)

    def act_forward(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False, clip_action: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        with th.no_grad():
            obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
            features, _ = self._get_features(obs_tensor_dict)
            distribution, _, _ = self._get_action_dist_from_features(features)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)

        actions = actions.cpu().numpy()
        actions = self.unscale_action(actions)
        if clip_action:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        log_prob = log_prob.cpu().numpy()

        return actions, log_prob


# Criticクラス
class Critic(nn.Module):
    def __init__(self, action_space, features_extractor, value_head_arch, device='cpu'):
        super(Critic, self).__init__()
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.value_head_arch = value_head_arch
        self.device = device
        self.activation_fn = nn.ReLU

        self.critic1 = self._build()
        self.critic2 = self._build()

    # Criticネットワークの構築
    def _build(self):
        last_layer_dim = self.features_extractor.features_dim + self.action_space.shape[0]
        critic_net = []
        for layer_size in self.value_head_arch:
            critic_net.append(nn.Linear(last_layer_dim, layer_size))
            critic_net.append(self.activation_fn())
            last_layer_dim = layer_size

        critic_net.append(nn.Linear(last_layer_dim, 1))  # Q-value output layer
        return nn.Sequential(*critic_net).to(self.device)

    # Q値を計算する関数
    def q_forward(self, obs_dict: Dict[str, np.ndarray], actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
        #状態から特徴量を抽出
        features, _ = self._get_features(obs_tensor_dict)
        actions = self.scale_action(actions)  
        #状態と行動を結合
        q_input = torch.cat([features, actions], dim=-1)
        # Q1、Q2値を計算
        q1 = self.critic1(q_input)
        q2 = self.critic2(q_input)
        return q1, q2


# SACクラス
class SacPolicy(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 policy_head_arch=[256, 256],
                 value_head_arch=[256, 256],
                 features_extractor_entry_point=None,
                 features_extractor_kwargs={},
                 distribution_entry_point=None,
                 distribution_kwargs={},
                 fake_birdview=False,
                 rgb_gail=False,
                 traj_plot=False,
                 gan_batches_done=0):
        super(SacPolicy, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor_entry_point = features_extractor_entry_point
        self.features_extractor_kwargs = features_extractor_kwargs
        self.distribution_entry_point = distribution_entry_point
        self.distribution_kwargs = distribution_kwargs
        self.fake_birdview = fake_birdview
        self.rgb_gail = rgb_gail
        self.traj_plot = traj_plot
        self.ortho_init = False
        # self.net_arch = [dict(pi=[256, 128, 64], vf=[128, 64])]
        self.policy_head_arch = list(policy_head_arch)
        self.value_head_arch = list(value_head_arch)        
        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'    

        features_extractor_class = load_entry_point(features_extractor_entry_point)
        if self.rgb_gail:
            self.features_extractor = features_extractor_class(observation_space, traj_plot=self.traj_plot, **features_extractor_kwargs)
        else:
            self.features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)
        distribution_class = load_entry_point(distribution_entry_point)

        self.action_dist = distribution_class(int(np.prod(action_space.shape)), **distribution_kwargs)

        
        if 'StateDependentNoiseDistribution' in distribution_entry_point:
            self.use_sde = True
            self.sde_sample_freq = 4
        else:
            self.use_sde = False
            self.sde_sample_freq = None

        # Actorモデルの作成
        self.actor_model = Actor(action_space, features_extractor, policy_head_arch, action_dist, device='cpu')
        # Criticモデルの作成
        self.critic_model = Critic(action_space, features_extractor, value_head_arch, device='cpu')

        # パラメータの初期化
        if self.ortho_init:
            self._init_weights(self.actor_model, gain=np.sqrt(2))
            self._init_weights(self.critic_model.critic1, gain=np.sqrt(2))
            self._init_weights(self.critic_model.critic2, gain=np.sqrt(2))
        
        # オプティマイザの設定
        self.actor_kwargs = {'lr': 3e-4, 'eps': 1e-5}
        self.critic_kwargs = {'lr': 3e-4, 'eps': 1e-5}
        self.actor_optimizer = th.optim.Adam(self.parameters(), **self.actor_kwargs)
        self.critic_optimizer = th.optim.Adam(self.parameters(), **self.critic_kwargs)

        if self.fake_birdview:
            self.gan_fake_birdview = GanFakeBirdview(gan_batches_done, traj_plot)
            
    def reset_noise(self, n_envs: int = 1) -> None:
        assert self.use_sde, 'reset_noise() is only available when using gSDE'
        self.action_dist.sample_weights(self.dist_sigma, batch_size=n_envs)
    
    # RGB画像またはBEV画像から特徴抽出をする関数（修正なし）
    def _get_features(self, obs_dict, fake_birdview=None) -> th.Tensor:
        """
        :param birdview: th.Tensor (num_envs, frame_stack*channel, height, width)
        :param state: th.Tensor (num_envs, state_dim)
        """
        state = obs_dict['state']

        if self.fake_birdview:
            if fake_birdview is None:
                with th.no_grad():
                    rgb_array = []
                    for rgb_key in ['central_rgb', 'left_rgb', 'right_rgb']:
                        rgb_img = (obs_dict[rgb_key].float() - 127.5) / 127.5
                        rgb_img = self.gan_fake_birdview.transforms(rgb_img)
                        rgb_array.append(rgb_img)

                    if self.traj_plot:
                        traj_plot = obs_dict["traj_plot"].float() / 255.0
                        rgb_array.append(traj_plot)
                    rgb = th.cat(rgb_array, dim=1)
                    cmd = obs_dict['cmd'].float()
                    traj = obs_dict['traj'].float()
                    fake_birdview = self.gan_fake_birdview.generator(rgb, cmd, traj)
                    fake_birdview = (fake_birdview * 127.5) + 127.5

            fake_birdview_clone = fake_birdview.clone().float() / 255.0
            features = self.features_extractor(fake_birdview_clone, state)
        elif self.rgb_gail:
            fake_birdview = obs_dict['birdview']
            rgb_array = []
            for rgb_key in ['central_rgb', 'left_rgb', 'right_rgb']:
                rgb_img = obs_dict[rgb_key].float() / 255.0
                rgb_array.append(rgb_img)

            if self.traj_plot:
                traj_plot = obs_dict["traj_plot_rgb"].float() / 255.0
                rgb_array.append(traj_plot)
            rgb = th.cat(rgb_array, dim=1)
            cmd = obs_dict['cmd'].float()
            traj = obs_dict['traj'].float()
            features = self.features_extractor(rgb, cmd, traj, state)

        else:
            fake_birdview = obs_dict['birdview']
            birdview = obs_dict['birdview'].float() / 255.0
            features = self.features_extractor(birdview, state)

        return features, fake_birdview


    # 特徴から行動分布,平均(mu)と標準偏差(sigma)のパラメータを返す（修正なし）
    def _get_action_dist_from_features(self, features: th.Tensor):
        latent_pi = self.actor(features)
        mu = self.dist_mu(latent_pi)
        if isinstance(self.dist_sigma, nn.Parameter):
            sigma = self.dist_sigma
        else:
            sigma = self.dist_sigma(latent_pi)
        return self.action_dist.proba_distribution(mu, sigma), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()


    # 行動における損失の計算に必要な情報を返す（一部修正）
    def evaluate_actions(self, obs_dict: Dict[str, th.Tensor], actions: th.Tensor, exploration_suggests, fake_birdview=None,):
        features, _ = self._get_features(obs_dict, fake_birdview)
        distribution, mu, sigma = self._get_action_dist_from_features(features)
        actions = self.scale_action(actions)
        log_prob = distribution.log_prob(actions)

        return distribution.entropy_loss(), distribution.exploration_loss(exploration_suggests)


    # BC損失の計算時に必要な情報を返す(一部修正)
    def evaluate_actions_bc(self, obs_dict: Dict[str, th.Tensor], fake_birdview, actions: th.Tensor):
        features, _ = self._get_features(obs_dict, fake_birdview)
        distribution, _, _ = self._get_action_dist_from_features(features)
        actions = self.scale_action(actions)
        log_prob = distribution.log_prob(actions)

        return log_prob, distribution.entropy_loss()


    # （一部修正：valuesを消した）
    def forward(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False, clip_action: bool = False):
        with th.no_grad():
            obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
            features, fake_birdview = self._get_features(obs_tensor_dict)
            distribution, mu, sigma = self._get_action_dist_from_features(features)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)

        actions = actions.cpu().numpy()
        actions = self.unscale_action(actions)
        if clip_action:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        log_prob = log_prob.cpu().numpy()
        features = features.cpu().numpy()
        fake_birdview = fake_birdview.cpu().numpy()

        return actions, log_prob, mu, sigma, features, fake_birdview


    # 行動を正規化した状態で返す（修正なし）
    def scale_action(self, action: th.Tensor, eps=1e-7) -> th.Tensor:
        # input action \in [a_low, a_high]
        # output action \in [d_low+eps, d_high-eps]
        d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

        if d_low is not None and d_high is not None:
            a_low = th.as_tensor(self.action_space.low.astype(np.float32)).to(action.device)
            a_high = th.as_tensor(self.action_space.high.astype(np.float32)).to(action.device)
            action = (action-a_low)/(a_high-a_low) * (d_high-d_low) + d_low
            action = th.clamp(action, d_low+eps, d_high-eps)
        return action


    # 行動を元の状態に直した状態で返す（修正なし）
    def unscale_action(self, action: np.ndarray, eps=0.0) -> np.ndarray:
        # input action \in [d_low, d_high]
        # output action \in [a_low+eps, a_high-eps]
        d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

        if d_low is not None and d_high is not None:
            # batch_size = action.shape[0]
            a_low, a_high = self.action_space.low, self.action_space.high
            # same shape as action [batch_size, action_dim]
            # a_high = np.tile(self.action_space.high, [batch_size, 1])
            action = (action-d_low)/(d_high-d_low) * (a_high-a_low) + a_low
            # action = np.clip(action, a_low+eps, a_high-eps)
        return action


    # モデルの初期パラメータを取得（SACの変更に伴い一部修正）
    def get_init_kwargs(self) -> Dict[str, Any]:
        if self.fake_birdview:
            gan_batches_done=self.gan_fake_birdview.batches_done
        else:
            gan_batches_done = 0

        init_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            policy_head_arch=self.policy_head_arch,
            value_head_arch=self.value_head_arch,
            features_extractor_entry_point=self.features_extractor_entry_point,
            features_extractor_kwargs=self.features_extractor_kwargs,
            distribution_entry_point=self.distribution_entry_point,
            distribution_kwargs=self.distribution_kwargs,
            fake_birdview=self.fake_birdview,
            rgb_gail=self.rgb_gail,
            traj_plot=self.traj_plot,
            gan_batches_done=gan_batches_done
        )

        return init_kwargs


    # 保存されたモデルを読み込む
    @classmethod
    def load(cls, path):
        if th.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables['policy_init_kwargs'])
        # Load weights
        model.load_state_dict(saved_variables['policy_state_dict'])
        model.to(device)
        return model, saved_variables['train_init_kwargs']


    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

