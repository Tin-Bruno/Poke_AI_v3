# Poke AI v3

Base para treinar uma IA a jogar Pokemon Red/Blue com reinforcement learning.
O projeto usa PyBoy como emulador de Game Boy, Gymnasium como interface de ambiente
e Stable-Baselines3 para treinar PPO.

> Importante: este repositorio nao inclui ROMs, saves ou assets comerciais. Use apenas
> uma ROM extraida legalmente de um cartucho que voce possui.

Este projeto segue a mesma linha do
[PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments),
especialmente a ideia da V2: recompensar exploracao por coordenadas e alimentar a
politica com tela, memoria do jogo e sinais de progresso.

## Ideia

"Zerar Pokemon" em uma unica politica do zero e um problema grande demais para atacar
direto. O caminho mais realista e montar um curriculo:

1. sair do quarto e chegar no laboratorio;
2. pegar o starter;
3. vencer o rival inicial;
4. explorar ate Viridian/Pewter;
5. vencer o Brock;
6. repetir com novos estados iniciais ate fechar ginasios, HMs, Elite Four e campeao.

Este starter kit entrega o primeiro bloco tecnico: um ambiente treinavel que observa a
tela, aperta botoes e recompensa exploracao, mapas novos, crescimento do time e badges.

## Setup

No PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Ou, se preferir instalar por `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Crie as pastas locais:

```powershell
New-Item -ItemType Directory -Force roms, states, runs
```

Edite o `.env` se quiser mudar caminhos e parametros padrao:

```text
POKE_ROM_PATH=roms/pokemon_red.gb
POKE_WINDOW=null
POKE_OBSERVATION_MODE=multi
POKE_TIMESTEPS=100000
```

Coloque sua ROM em algo como:

```text
roms/pokemon_red.gb
```

## Testar o ambiente

```powershell
python scripts/check_env.py --rom .\roms\pokemon_red.gb --steps 200
```

Ou, se `POKE_ROM_PATH` ja estiver certo no `.env`:

```powershell
python scripts/check_env.py --steps 200
```

Se voce tiver um arquivo de simbolos do projeto pret/pokered, passe tambem:

```powershell
python scripts/check_env.py --rom .\roms\pokemon_red.gb --symbols .\roms\pokered.sym
```

Sem simbolos, o projeto usa enderecos comuns de Pokemon Red/Blue vanilla para ler mapa,
coordenadas, badges e niveis do time.

## Treinar

Treino inicial curto:

```powershell
python scripts/train_ppo.py --rom .\roms\pokemon_red.gb --timesteps 100000
```

Com `.env` configurado:

```powershell
python scripts/train_ppo.py
```

Por padrao o treino usa `POKE_OBSERVATION_MODE=multi`, que escolhe `MultiInputPolicy`
e entrega ao modelo:

- `screens`: ultimos frames da tela em 72x80;
- `health`: fracao de HP do time;
- `level`: soma dos niveis codificada em Fourier;
- `badges`: oito bits de insignias;
- `events`: flags de progresso do jogo;
- `map`: mapa local de coordenadas visitadas;
- `recent_actions`: historico curto de acoes.

Treino maior:

```powershell
python scripts/train_ppo.py --timesteps 5000000 --n-envs 4 --vec-env subproc
```

Os checkpoints ficam em `runs/checkpoints/` e o modelo final em `runs/models/`.

## Assistir a IA

```powershell
python scripts/eval_agent.py --rom .\roms\pokemon_red.gb --model .\runs\models\poke_red_ppo.zip --window SDL2
```

Se `SDL2` der problema no Windows, rode com janela nula:

```powershell
python scripts/eval_agent.py --rom .\roms\pokemon_red.gb --model .\runs\models\poke_red_ppo.zip --window null
```

## Usar estados iniciais

Para curriculo, salve estados do PyBoy em marcos importantes, por exemplo:

```text
states/01-bedroom.state
states/02-oaks-lab.state
states/03-before-brock.state
```

Depois treine a partir de um estado:

```powershell
python scripts/train_ppo.py --rom .\roms\pokemon_red.gb --state .\states\02-oaks-lab.state --timesteps 1000000
```

## Ajustes que mais importam

- `POKE_OBSERVATION_MODE`: `multi` segue a ideia do PokemonRedExperiments V2; `screen`
  treina apenas em pixels com `CnnPolicy`.
- `--action-frames`: quantos frames cada acao dura. Valores entre 8 e 24 costumam ser bons.
- `POKE_MAX_STEPS`: tamanho maximo de cada episodio. O default e `2048 * 80`.
- `--max-no-progress-steps`: encerra episodio quando a IA para de explorar.
- `POKE_REWARD_SCALE` e `POKE_EXPLORE_WEIGHT`: controlam o peso global e o peso de exploracao.
- `RewardConfig` em `src/poke_ai_v3/rewards.py`: pesos de recompensa por mapa, posicao,
  eventos, cura, badge, nivel e tamanho do time.
- Estados iniciais: sao mais importantes que mexer no algoritmo no comeco.

## Proximos passos recomendados

1. Validar que o agente sai de telas de menu com recompensas simples.
2. Criar estados iniciais por fase.
3. Adicionar recompensas especificas por objetivo, como pegar Pokedex, obter Cut/Surf e vencer Elite Four.
4. Registrar videos dos melhores episodios para entender onde a IA trava.
