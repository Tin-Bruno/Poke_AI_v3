__all__ = ["FreeplayConfig", "FreeplayPokemonRedEnv", "PokemonRedEnv"]


def __getattr__(name: str):
    if name == "PokemonRedEnv":
        from envs.pokemon_red_env import PokemonRedEnv

        return PokemonRedEnv
    if name in {"FreeplayConfig", "FreeplayPokemonRedEnv"}:
        from envs.freeplay_env import FreeplayConfig, FreeplayPokemonRedEnv

        return {"FreeplayConfig": FreeplayConfig, "FreeplayPokemonRedEnv": FreeplayPokemonRedEnv}[name]
    raise AttributeError(f"module 'envs' has no attribute {name!r}")
