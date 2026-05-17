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

## Testar a ROM Pura

Antes de culpar o ambiente ou os states, rode a ROM direto no PyBoy:

```bash
python scripts/test_rom.py --window SDL2 --mode native --seconds 180
```

Esse modo nao carrega state e nao usa o ambiente Gym. Clique na janela do PyBoy e
jogue como em um emulador normal. Se o dialogo tambem travar aqui, o problema pode
estar na ROM, SRAM ou instalacao do PyBoy.

Para investigar bug de caixa de dialogo, este e o teste mais importante:

```bash
python scripts/test_rom.py --window SDL2 --mode native --seconds 0
```

Ele usa renderizacao nativa do PyBoy, sem `window="null"` e sem `tick(..., False)`.
Feche com `Ctrl+C` no terminal.

Para confirmar se a tela tem caixa de dialogo desenhada:

```bash
python scripts/dialog_probe.py --phase phase1 --window SDL2
```

Ele imprime algo assim:

```text
dialog_visual=True score=0.532 white=0.612 dark=0.284 wx=-7 wy=0 tiles=18 map=0 x=6 y=0
```

No `manual_control.py`, tambem da para digitar:

```text
dialog
```

Campos importantes:

```text
dialog_visual=True   caixa de texto detectada pela imagem
white alto           a parte inferior tem fundo claro, normal em textbox
dark alto e white baixo  a parte inferior esta preta, nao e textbox
wx/wy                posicao da window layer do Game Boy
tiles                variedade de tiles na parte inferior
```

Se quiser salvar a imagem que o detector esta vendo:

```bash
python scripts/dialog_probe.py --phase phase1 --window SDL2 --seconds 5 --screenshot logs/dialog_probe.png
```

No `manual_control.py`, o comando abaixo salva uma screenshot:

```text
screen logs/manual_screen.png
```

Se `dialog_visual=False`, a caixa de texto nao esta desenhada na tela. Se o jogo
parece travado mesmo assim, provavelmente ele esta aguardando algum evento/menu,
mas nao em uma textbox visual.

Para testar os botoes injetados pelo terminal, sem state:

```bash
python scripts/test_rom.py --window SDL2 --mode scripted
```

No modo scripted:

```text
u        START
talk 20  aperta A com pausa de dialogo
wait 60  espera 60 frames
save logs/rom_test.state
q        sair
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

Metodo recomendado, sem bug de input do terminal:

```bash
python scripts/save_state.py --phase phase1
```

Jogue direto na janela do PyBoy. Quando estiver no ponto certo, feche a janela; o
script salva `states/phase1_start.state`.

Metodo de debug com botoes pelo terminal:

```bash
python scripts/manual_control.py --phase phase1 --window SDL2
```

Comandos do manual:

```text
w/a/s/d mover
j=A, k=B, u=START, i=SELECT
p imprimir RAM
dialog confirma se ha caixa de dialogo na tela
screen logs/manual_screen.png salva uma screenshot
talk 10 avanca dialogo apertando A com pausa
wait 60 espera 60 frames
save states/phase1_start.state
q sair
```

Para nao apertar Enter em cada botao, digite uma sequencia e aperte Enter uma vez:

```text
jjjj
ddddww
10j
5d
```

No Git Bash esse modo costuma ser melhor que captura de tecla unica, porque evita
fila gigante de tecla repetida no terminal.

Para dialogos, prefira `talk` em vez de `10j`, porque ele espera mais entre cada A:

```text
talk 10
talk 30
```

Se ainda quiser testar tecla unica sem Enter:

```bash
python scripts/manual_control.py --phase phase1 --window SDL2 --fast-mode
```

Se os toques ficarem curtos ou longos demais, ajuste os frames:

```bash
python scripts/manual_control.py --phase phase1 --window SDL2 --action-frames 8
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
- `scripts/save_state.py`: jogar direto na janela PyBoy e salvar state ao fechar.
- `scripts/test_rom.py`: testar a ROM pura, sem carregar state nem ambiente Gym.
- `scripts/dialog_probe.py`: monitorar se a caixa de dialogo esta desenhada.
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
