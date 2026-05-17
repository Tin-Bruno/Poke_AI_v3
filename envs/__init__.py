__all__ = ["PokemonRedEnv"]


def __getattr__(name: str):
    if name == "PokemonRedEnv":
        from envs.pokemon_red_env import PokemonRedEnv

        return PokemonRedEnv
    raise AttributeError(f"module 'envs' has no attribute {name!r}")
