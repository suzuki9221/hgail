# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/models/ppo_buffer.py

from gym import spaces
import numpy as np
from typing import Optional, Generator, NamedTuple, Dict, List
import torch as th
from stable_baselines3.common.vec_env.base_vec_env import tile_images
import cv2
import time
from threading import Thread
import queue


class SacBufferSamples(NamedTuple):
    observations: Dict[str, th.Tensor]       # 現在の状態
    actions: th.Tensor                       # 行動
    rewards: th.Tensor                        # 報酬
    next_observations: Dict[str, th.Tensor]  # 次の状態
    dones: th.Tensor                         # 終了条件(True or False)
    fake_birdviews: th.Tensor                # BEV画像
    exploration_suggests: List[tuple]        # アクセル、ステアリングに関する情報


class SacBuffer():
    # 初期化
    def __init__(self, buffer_size: int,  sample_size: int,observation_space: spaces.Space, action_space: spaces.Space,
                 gamma: float = 0.99, n_envs: int = 1):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma                # 割引率（ターゲットのQ値）
        self.n_envs = n_envs              # 並列数
        self.reset()
        self.pos = 0                      # 現在の並列の添字
        self.full = False        
        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.sample_queue = queue.Queue()
    

    # 初期化
    def reset(self) -> None:
        self.observations = {}
        for k, s in self.observation_space.spaces.items():
            self.observations[k] = np.zeros((self.buffer_size, self.n_envs,)+s.shape, dtype=s.dtype)
        self.next_observations = {}
        for k, s in self.observations_space.spaces.items():
            self.next_observations[k] = np.zeros((self.buffer_size, self.n_envs,)+s.shape, dtype=s.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs)+self.action_space.shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.exploration_suggests = np.zeros((self.buffer_size, self.n_envs), dtype=[('acc', 'U10'), ('steer', 'U10')])
        s = self.observation_space.spaces['birdview']
        self.fake_birdviews = np.zeros((self.buffer_size, self.n_envs,)+s.shape, dtype=s.dtype)
        self.reward_debugs = [[] for i in range(self.n_envs)]
        self.terminal_debugs = [[] for i in range(self.n_envs)]

        self.pos = 0
        self.full = False

    # リプレイバッファにデータを追加
    def add(self,
            obs_dict: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            next_obs_dict: Dict[str, np.ndarray],
            done: np.ndarray,
            fake_birdview: np.ndarray,
            infos) -> None:

        for k, v in obs_dict.items():
            self.observations[k][self.pos] = v
        self.actions[self.pos] = action
        for k, v in next_obs_dict.items():
            self.next_observations[k][self.pos] = v
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.fake_birdviews[self.pos] = fake_birdview

        for i in range(self.n_envs):
            self.reward_debugs[i].append(infos[i]['reward_debug']['debug_texts'])
            self.terminal_debugs[i].append(infos[i]['terminal_debug']['debug_texts'])

            n_steps = infos[i]['terminal_debug']['exploration_suggest']['n_steps']
            if n_steps > 0:
                n_start = max(0, self.pos-n_steps)
                self.exploration_suggests[n_start:self.pos, i] = \
                    infos[i]['terminal_debug']['exploration_suggest']['suggest']

        self.pos += 1

        # バッファサイズがフルになったら、先頭に戻る
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    
    # バッチサイズの大きさでデータを取得する関数
    def get(self, batch_size: Optional[int] = None) -> Generator[SacBufferSamples, None, None]:
        if self.full:
            all_buffer_size = self.buffer_size * self.n_envs
        else: 
            all_buffer_size = self.pos * self.n_envs
        assert all_buffer_size >= self.sample_size, f"Insufficient Buffer size: {all_buffer_size} available, {self.sample_size} needed"

        indices = np.random.permutation(all_buffer_size)[:self.sample_size]

        # Prepare the data
        for tensor in ['actions', 'rewards', 'dones', 'fake_birdviews', 'exploration_suggests']:
            self.__dict__['flat_'+tensor] = self.flatten(self.__dict__[tensor])
        self.flat_observations = {}
        for k in self.observations.keys():
            self.flat_observations[k] = self.flatten(self.observations[k])
        self.flat_next_observations = {}
        for k in self.next_observations.keys():
            self.flat_next_observations[k] = self.flatten(self.next_observations[k])

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.sample_size 

        start_idx = 0
        while start_idx < self.sample_size:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    
    # sac_buffer.py内のget関数のみで用いられる
    # SacBufferSamplesの構造でデータサンプルを取得する関数
    def _get_samples(self, batch_inds: np.ndarray) -> SacBufferSamples:
        def to_torch(x):
            return th.as_tensor(x).to(self.device)
            # return th.from_numpy(x.astype(np.float32)).to(self.device)

        obs_dict = {}
        for k in self.observations.keys():
            obs_dict[k] = to_torch(self.flat_observations[k][batch_inds])
        next_obs_dict = {}
        for k in self.next_observations.keys():
            next_obs_dict[k] = to_torch(self.flat_next_observations[k][batch_inds])
        actions_torch = to_torch(self.flat_actions[batch_inds])
        rewards_torch = to_torch(self.flat_rewards[batch_inds])
        dones_torch = to_torch(self.flat_dones[batch_inds])
        fake_birdviews_torch = to_torch(self.flat_fake_birdviews[batch_inds])

        data_torch = (
            obs_dict, 
            actions_torch, 
            rewards_torch, 
            next_obs_dict,
            dones_torch,
            fake_birdviews_torch,
            self.flat_exploration_suggests[batch_inds]
        )
        
        return SacBufferSamples(*data_torch)


    # get関数で用いられる
    # 並列環境の情報を結合
    @staticmethod
    def flatten(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape

        return arr.reshape(shape[0] * shape[1], *shape[2:])

    
    # BEV画像に情報を付与
    def render(self):
        assert self.exceeded, ''
        list_render = []

        _, _, c, h, w = self.observations['birdview'].shape
        vis_idx = np.array([0, 1, 2])

        for i in range(self.buffer_size): #修正すべき
            im_envs = []
            for j in range(self.n_envs):
                im_birdview = self.observations['birdview'][i, j, :, :, :]
                im_birdview = np.transpose(im_birdview, [1, 2, 0]).astype(np.uint8)

                im_fake_birdview = self.fake_birdviews[i, j, :, :, :]
                im_fake_birdview = np.transpose(im_fake_birdview, [1, 2, 0]).astype(np.uint8)

                im = np.zeros([h, w*3, 3], dtype=np.uint8)
                im[:h, :w] = im_birdview
                im[:h, w:2*w] = im_fake_birdview

                action_str = np.array2string(self.actions[i, j], precision=1, separator=',', suppress_small=True)
                state_str = np.array2string(self.observations['state'][i, j],
                                            precision=1, separator=',', suppress_small=True)
                next_state_str = np.array2string(self.next_observations['state'][i, j],
                                            precision=1, separator=',', suppress_small=True)
                reward = self.rewards[i, j]
                done = int(self.dones[i, j])

                txt_1 = f'a:{action_str}'
                im = cv2.putText(im, txt_1, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                txt_2 = f'{done} {state_str} {next_state_str}'
                im = cv2.putText(im, txt_2, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                txt_3 = f'rw:{reward:5.2f} next_s:{next_state_str}'
                im = cv2.putText(im, txt_3, (2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                for i_txt, txt in enumerate(self.reward_debugs[j][i] + self.terminal_debugs[j][i]):
                    im = cv2.putText(im, txt, (2*w, (i_txt+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                im_envs.append(im)

            big_im = tile_images(im_envs)
            list_render.append(big_im)

        return list_render

    # GPUにバッチサイズのデータを送るための準備
    def start_caching(self, batch_size):
        thread1 = Thread(target=self.cache_to_cuda, args=(batch_size,))
        thread1.start()

    
    # GPUにデータを送る
    def cache_to_cuda(self, batch_size):
        self.sample_queue.queue.clear()

        for rollout_data in self.get(batch_size):
            while self.sample_queue.qsize() >= 2:
                time.sleep(0.01)
            self.sample_queue.put(rollout_data)

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos
