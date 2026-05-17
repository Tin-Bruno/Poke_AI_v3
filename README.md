# Pokemon Red AI

Projeto limpo para treinar uma IA a jogar Pokemon Red por fases, usando PyBoy,
Gymnasium e Stable-Baselines3.

> Este repositorio nao inclui ROMs, saves ou assets comerciais. Use apenas uma ROM
> extraida legalmente de um cartucho que voce possui.

## Estrutura

```text
roms/        ROM local, ignorada pelo git
states/      save states por fase, ignorados pelo git
models/      modelos treinados por fase, ignorados pelo git
memory/      mapa de RAM e leitura do estado do jogo
envs/        ambiente Gym, step handler e success conditions
rewards/     rewards pequenas e combinaveis
phases/      configuracao declarativa das fases
scripts/     treino, avaliacao e ferramentas manuais
logs/        saidas locais
runs/        checkpoints e TensorBoard
```

A regra principal: o ambiente nao sabe a logica de cada fase. Ele executa o jogo,
le RAM, aplica a acao e chama a configuracao da fase atual.

## Setup

No Git Bash:

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p roms states models logs runs
```

O `python --version` deve mostrar Python 3.11.x. Se voce usa `python -m venv`
direto, confira antes se o `python` do terminal aponta para 3.11.

Alternativa no PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force roms, states, models, logs, runs
```

Evite criar o venv com Python 3.14 por enquanto; `torch` e `stable-baselines3`
costumam ser mais estaveis em 3.11/3.12.

Se o ambiente ja existe com a versao errada:

```bash
deactivate
rm -rf .venv
py -3.11 -m venv .venv
source .venv/Scripts/activate
python --version
python -m pip install -r requirements.txt
```

Forma generica, caso seu `python` ja seja 3.11:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p roms states models logs runs
```

Coloque sua ROM em:

```text
roms/pokemon_red.gb
```

Edite o `.env` se precisar:

```text
POKE_ROM_PATH=roms/pokemon_red.gb
POKE_PHASE=phase1
POKE_STATES_DIR=states
POKE_OBSERVATION_MODE=multi
```

## Fases

As fases ficam em [phases/phase_config.py](phases/phase_config.py).

Comecamos com:

```text
phase1  sair do quarto
phase2  sair da casa
phase3  ativar evento do Professor Oak
phase4  ser levado ao laboratorio
phase5  escolher starter
phase6  passar dialogo do rival
phase7  vencer primeira batalha
phase8  sair do laboratorio
phase9  ir para Rota 1
```

O primeiro alvo real e deixar `phase1` e `phase2` funcionando muito bem antes de
avancar.

## Fluxo Recomendado

1. Criar/conferir o state da fase manualmente:

```bash
python scripts/manual_control.py --phase phase1 --window SDL2
```

Comandos do manual:

```text
w/a/s/d mover
j=A, k=B, u=START, i=SELECT
p imprimir RAM
save states/phase1_start.state
q sair
```

2. Testar o ambiente:

```bash
python scripts/test_env.py --phase phase1 --steps 200
```

3. Treinar a fase:

```bash
python scripts/train_phase.py --phase phase1 --timesteps 100000
```

4. Avaliar:

```bash
python scripts/eval_phase.py --phase phase1 --window SDL2
```

5. Se a fase passar, salvar o state para a proxima:

```bash
python scripts/eval_phase.py --phase phase1 --save-success-state states/phase2_start.state
```

## Scripts

- `scripts/manual_control.py`: jogar manualmente, ver `map/x/y`, salvar states.
- `scripts/save_state.py`: atalho para o controle manual focado em salvar states.
- `scripts/test_env.py`: validar Gym/env de uma fase.
- `scripts/train_phase.py`: treinar uma fase especifica.
- `scripts/eval_phase.py`: rodar um modelo treinado.
- `scripts/eval_sequence.py`: esqueleto para rodar fases em sequencia.
- `scripts/view_ram.py`: imprimir snapshot de RAM de um state.
- `scripts/visualization.py`: listar fases configuradas.

## Onde Vamos Comecar

Primeiro vamos criar um `states/phase1_start.state` confiavel. Depois vamos treinar
somente `phase1`, que termina quando o `map_id` vira `39`. Quando isso estiver
estavel, salvamos `states/phase2_start.state` e repetimos o processo para sair da
casa.
