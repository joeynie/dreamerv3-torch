"""Simple evaluator for DreamerV3-trained agents.

Usage examples:
    python evaluate.py --config portal2_vision --logdir ./logdir/portal2_basic --episodes 3

This script will:
- load the config named by --config from configs.yaml
- instantiate the same environment used for training via make_env
- load checkpoint `latest.pt` (or --checkpoint path)
- run `--episodes` episodes, capture frames and save MP4s to <logdir>/eval_videos
- print per-episode and average rewards

"""

import argparse
import pathlib
import ruamel.yaml as yaml
import torch
import numpy as np
import imageio
import sys
import os
from types import SimpleNamespace
import functools
from parallel import Damy

# Ensure importing dreamer and tools from this folder
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dreamer import Dreamer, make_env
import tools


def load_config(name):
    configs = yaml.safe_load((ROOT / "configs.yaml").read_text())
    # merge defaults + named config
    defaults = configs.get("defaults", {})
    target = configs.get(name)
    if target is None:
        raise ValueError(f"Config {name} not found in configs.yaml")
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    merged = dict(defaults)
    recursive_update(merged, target)
    merged["size"] = tuple(merged["size"])
    return SimpleNamespace(**merged)


def load_agent(checkpoint_path, obs_space, act_space, config, device, logger):
    agent = Dreamer(obs_space, act_space, config, logger, None).to(device)
    agent.requires_grad_(False)
    if checkpoint_path and pathlib.Path(checkpoint_path).exists():
        ck = torch.load(checkpoint_path, map_location=device)
        if "agent_state_dict" in ck:
            agent.load_state_dict(ck["agent_state_dict"]) 
        else:
            agent.load_state_dict(ck)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Warning: checkpoint not found, using randomly initialized agent")
    agent.eval()
    return agent


def obs_to_tensors(obs, device):
    # obs can be dict or numpy array. Agent expects tensors in a dict form.
    if isinstance(obs, tuple) or isinstance(obs, list):
        obs = obs[0]
    if isinstance(obs, np.ndarray):
        # single image
        return {"image": torch.from_numpy(obs).unsqueeze(0).to(device)}
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                v = np.ascontiguousarray(v)
                out[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            elif isinstance(v, (int, float, bool)):
                out[k] = torch.tensor([v]).to(device)
            else:
                try:
                    out[k] = torch.tensor(v).unsqueeze(0).to(device)
                except Exception:
                    pass
        return out
    return {}


def run_eval(args):
    config = load_config(args.config)
    # config.logdir = pathlib.Path(args.logdir)
    logdir = pathlib.Path(config.logdir).expanduser()
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Create eval environment
    env = make_env(config, "eval", 0)
    acts = env.action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    # instantiate agent
    checkpoint = args.checkpoint or (pathlib.Path(config.logdir) / "latest.pt")
    logger = tools.Logger(logdir, args.episodes * args.max_steps)
    device = torch.device(config.device)
    agent = load_agent(checkpoint, env.observation_space, env.action_space, config, device, logger)

    outdir = logdir / "eval_videos"
    outdir.mkdir(parents=True, exist_ok=True)

    # Use tools.simulate to run evaluation similar to dreamer.py
    # Wrap single env with Damy to match simulate's expected env wrapper interface
    eval_envs = [Damy(env)]
    # load any existing episodes (simulate will append to this cache and save to config.evaldir)
    eval_cache = tools.load_episodes(config.evaldir, limit=0)
    eval_policy = functools.partial(agent, training=False)

    # Run simulation: this will populate files under config.evaldir
    try:
        tools.simulate(
            eval_policy,
            eval_envs,
            eval_cache,
            config.evaldir,
            logger,
            is_eval=True,
            episodes=args.episodes
        )
    finally:
        # Whether simulate finished normally or was interrupted, try to export
        # on-disk episodes first
        saved = tools.load_episodes(config.evaldir, limit=args.episodes)
        # Also include any in-memory episodes from eval_cache that may not have been flushed
        # Merge eval_cache entries not present in saved (preserve insertion order)
        merged = dict(saved)
        try:
            for k, v in eval_cache.items():
                if k not in merged:
                    merged[k] = v
        except Exception:
            # eval_cache may be None or not a mapping; ignore in that case
            pass

        rewards = []
        for i, (name, ep) in enumerate(list(merged.items())[: args.episodes]):
            ep_reward = float(np.sum(ep.get("reward", []))) if "reward" in ep else 0.0
            rewards.append(ep_reward)
            # extract frames if present
            if "image" in ep:
                frames = [np.asarray(f).astype(np.uint8) for f in ep["image"]]
                video_path = outdir / f"eval_ep_{i+1}.mp4"
                try:
                    writer = imageio.get_writer(str(video_path), fps=args.fps)
                    for f in frames:
                        arr = np.asarray(f)
                        if arr.ndim == 2:
                            arr = np.stack([arr] * 3, axis=-1)
                        if arr.dtype != np.uint8:
                            arr = (255 * (arr / arr.max() if arr.max() else arr)).astype(np.uint8)
                        writer.append_data(arr)
                    writer.close()
                    print(f"Saved video: {video_path} (frames={len(frames)})")
                except Exception as e:
                    print(f"Failed to save video: {e}")
            else:
                print(f"No image frames found for episode {name}")

            print(f"Episode {i+1} reward: {ep_reward:.3f}")

        if rewards:
            print(f"Average reward over {len(rewards)}: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        else:
            print("No evaluation episodes were produced.")

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config name in configs.yaml used for training")
    parser.add_argument("--checkpoint", default=None, help="optional specific checkpoint path")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()
    run_eval(args)
