from rewards.registry import PhaseReward, make_reward

__all__ = ["FreeplayReward", "PhaseReward", "make_reward"]


def __getattr__(name: str):
    if name == "FreeplayReward":
        from rewards.freeplay_reward import FreeplayReward

        return FreeplayReward
    raise AttributeError(f"module 'rewards' has no attribute {name!r}")
