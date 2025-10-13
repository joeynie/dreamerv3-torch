import numpy as np
import time
import yaml
import gymnasium as gym
from gymnasium.spaces import Box

# 环境相关导入
from interface.gameio.io_env import IOEnvironment
from interface.config import Config
from interface.gameio.lifecycle.ui_control import switch_pause_status
from interface.video import VideoRecordProvider
from il.replayer import InputReplayer
from rl.reward_manager import RewardManager
from rl.utils import get_current_frame, validate_list_action, execute_action
from rl.detector import Detector
from rl.monitor import Monitor

config = Config()

class CustomEnv(gym.Env):
    """自定义 Gym 环境，用于 AAA 游戏"""
    def __init__(self, game_name: str, max_episode_steps=4000, debug: bool = False):
        super().__init__()
        self.game_name = game_name
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        config.env_name = game_name
        config._set_env_window_info()
        self.io = IOEnvironment()
        self.action_space = Box(
            low=np.array([0]*9 + [-30, -30], dtype=np.float32),
            high=np.array([1]*9 + [30, 30], dtype=np.float32),
            dtype=np.float32
        )
        self.image_shape = (800, 600)
        self.observation_space = Box(low=0, high=255, shape=(self.image_shape[1], self.image_shape[0], 3), dtype=np.uint8)
        self.replayer = InputReplayer()
        self.video_recorder = VideoRecordProvider()
        self.detector = Detector()
        self.monitor = Monitor(detector=self.detector, io=self.io)

        # 初始化 reward manager
        self.mission = "Your mission is to move forward."
        with open("vlm/vlm_config.yaml", "r", encoding='utf-8') as f:
            config_yaml = yaml.safe_load(f)
            vlm_config = config_yaml.get("vlm_models", {})
            model_config = vlm_config.get("custom_qwen", {})
        with open("rl/rl_config.yaml", "r", encoding='utf-8') as f:
            config_yaml = yaml.safe_load(f)
            prompt_template = config_yaml.get("prompts", {}).get("reward_prompt", {})
        self.reward_manager = RewardManager(reward_mode="direct", model_configs=[model_config], prompt_template=prompt_template, monitor=self.monitor, debug=debug)

        self.render = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.monitor.reset()
        self.detector.reset()
        config.env_window.set_foreground()
        # 重启关卡
        self.video_recorder.set_pause(True)
        self.replayer.restart_level()
        # switch_pause_status()
        self.replayer.replay()
        frame = get_current_frame(config.env_region, size=self.image_shape)
        # switch_pause_status()
        self.video_recorder.set_pause(False)
        return frame, {}

    def _process_action(self, action):
        """把 SB3 输出的动作转为游戏可执行的动作"""
        if isinstance(action, np.ndarray):
            processed_action = np.array(action, dtype=np.float32)
            processed_action[:9] = (processed_action[:9] > 0.5).astype(np.int32)
            processed_action = validate_list_action(processed_action, self.action_space)
            # print(f"动作: {processed_action}")
            return processed_action
        return validate_list_action(action, self.action_space)

    def step(self, action):
        self.current_step += 1

        # switch_pause_status()
        action = self._process_action(action)
        execute_action(self.io, action)
        frame = get_current_frame(config.env_region, size=self.image_shape)
        # switch_pause_status()
        self.obs = self.detector.detect(frame, label_filter=self.monitor.get_cur_target())
        if self.obs is None: self.obs = np.zeros(3, dtype=np.float32)     
               
        # 计算奖励
        reward, is_task_complete = self.reward_manager.compute_reward(
            game_name=self.game_name,
            env_reward=0.0,
            frame=frame,
            mission=self.mission,
            action=action,
            obs=self.obs
        )
        self.monitor.check_miss()
        reach = self.monitor.check_reach(self.obs)
        if reach:
            reward += 5.0
            self.monitor.update_task(self.obs)
        terminated = self.monitor.get_cur_target() == None
        if terminated:
            reward += 10.0
        truncated = self.current_step >= self.max_episode_steps
        info = {}
        return frame, reward, terminated, truncated, info

    def close(self):
        self.io.release_held_keys()
        self.io.release_held_buttons()
        self.video_recorder.finish_capture()