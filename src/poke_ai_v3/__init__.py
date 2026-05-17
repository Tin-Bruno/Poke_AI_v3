"""Ferramentas para treinar agentes em Pokemon Red/Blue via PyBoy."""

__all__ = ["PokemonRedEnv"]


def __getattr__(name: str):
    if name == "PokemonRedEnv":
        from poke_ai_v3.env import PokemonRedEnv

        return PokemonRedEnv
    raise AttributeError(f"module 'poke_ai_v3' has no attribute {name!r}")
