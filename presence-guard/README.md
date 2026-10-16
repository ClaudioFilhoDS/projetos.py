# Presence Guard

Usa a câmera infravermelha do ThinkPad (a mesma que o Howdy usa) para:

- **borrar a tela** quando você não está olhando (padrão: 2,5 s), sem nenhum texto, em níveis:
  - **virou o rosto** → borrão **leve** (dá para ver as janelas, não dá para ler);
  - **outra pessoa olhando junto** → **médio** (com `outra_pessoa = borrar`);
  - **saiu** → **forte**, que vai **escurecendo aos poucos** até bloquear (uma contagem sem números);
- **bloquear o Ubuntu** se você sair e não voltar (padrão: 1 min borrado, depois bloqueia);
- **entende o teclado e o mouse**: digitando olhando o teclado não borra, e mexer o mouse tira o
  borrão leve;
- **reconhecer você com e sem óculos**: se outra pessoa sentar no seu lugar, borra na hora
  (ela não consegue tirar o borrão) e bloqueia em 15 s;
- avisar (ou borrar) quando **outra pessoa** está olhando a sua tela junto com você;
- opcional: quando você volta, **chamar o Howdy sozinho** para desbloquear sem apertar nada.

Funciona no escuro (IR), roda tudo localmente e não salva nenhuma imagem.
O `sudo` com o Howdy continua funcionando: antes do Howdy rodar, o programa solta a câmera.

Feito para: Ubuntu 24.04, sessão **Ubuntu on Xorg** (X11), Howdy já funcionando.

## Como funciona

1. **YuNet** (OpenCV, 230 KB) acha os rostos na imagem, até pequenos ou longe.
2. **MediaPipe** analisa só o recorte do seu rosto: para onde a cabeça e os olhos apontam.
3. **SFace** (OpenCV) compara o rosto com o seu cadastro (com e sem óculos).
4. De óculos o infravermelho reflete nas lentes; quando ele vê o reflexo (ou reconhece você
   de óculos), passa a usar só a direção da cabeça.
5. Virou tanto o rosto que ninguém mais acha o rosto (de perfil)? No infravermelho a sua cabeça
   perto da câmera fica clara e o fundo escuro: se onde estava o rosto continua claro, ele sabe
   que você só virou (borrão leve). Isso muda só o nível do borrão; o tempo para bloquear
   continua contando, por segurança.
6. Teclado/mouse (extensão XScreenSaver do X11): mexendo neles você conta como presente e
   olhando — menos quando outra pessoa está no seu lugar.

Para gastar pouco: analisa 1,5x por segundo quando está tudo normal e 4x por segundo só quando
algo muda (desviou o olhar, saiu, tela borrada). Na bateria, metade disso. Entre uma análise e
outra o programa dorme, e a tela de bloqueio é acompanhada por eventos, sem ficar perguntando.

## Instalar (ou atualizar)

```bash
sudo apt install -y git
git clone -b claude/upbeat-sagan-2jhkne https://github.com/ClaudioFilhoDS/projetos.py.git
cd projetos.py/presence-guard
./install.sh
```

Para atualizar uma instalação antiga: baixe a versão nova (ou `git pull`) e rode `./install.sh`
de novo. Se a sua config era da versão anterior, ela fica guardada em
`~/.config/presence-guard/config.ini.anterior`.

O instalador:

1. instala `python3-venv python3-tk libnotify-bin xdotool` (pede seu rosto/senha);
2. cria um ambiente Python isolado em `~/.local/share/presence-guard` e baixa os modelos (~43 MB);
3. coloca uma linha no PAM **antes** da linha do Howdy (`pam_exec ... presence-guard-yield`),
   com backup `*.bak-presence-guard`. Isso só avisa o programa para soltar a câmera; não
   muda a sua autenticação;
4. liga o serviço (inicia junto com a sua sessão);
5. oferece a calibração e o cadastro do rosto.

**Depois de instalar, teste o sudo em outro terminal** (deixe o do instalador aberto):

```bash
sudo -k; sudo echo funcionou
```

Se algo der errado, no terminal do instalador:
`for f in /etc/pam.d/*.bak-presence-guard; do sudo cp -a "$f" "${f%.bak-presence-guard}"; done`

## Cadastrar seu rosto (com e sem óculos)

```bash
presence-guard cadastrar
```

Ele pede para você ficar **sem óculos** e depois **com óculos** (digite `p` para pular se não
usa), uns 8 segundos cada. Durante a captura, olhe para a tela e mexa a cabeça devagar.
No final ele testa se te reconhece.

- `presence-guard cadastrar --adicionar` soma amostras novas (ex.: óculos novos, outra luz).
- Logo depois que você desbloqueia (Howdy ou senha), se ele não tinha te reconhecido bem, ele
  guarda uma amostra nova sozinho (`aprender_sozinho`).
- `presence-guard esquecer` apaga o cadastro (aí qualquer rosto volta a contar como você).

Sem cadastro tudo funciona igual, só não diferencia você de outra pessoa.

## Calibrar o olhar (recomendado)

```bash
presence-guard calibrate
```

Aparece um ponto no centro e nos 4 cantos da tela; siga com os olhos do jeito que você
normalmente usa o notebook. Leva uns 25 s. Refaça se mudar muito a posição em que você usa
o notebook. O serviço pega a calibração nova sozinho.

## Comandos

| comando | o que faz |
|---|---|
| `presence-guard off` | **desliga de verdade**: para o processo, solta a câmera, zero CPU. Continua desligado depois de reiniciar |
| `presence-guard on` | liga de novo (e volta a iniciar sozinho no login) |
| `presence-guard toggle` | alterna entre ligado/desligado, com notificação |
| `presence-guard cadastrar` | grava seu rosto com e sem óculos |
| `presence-guard calibrate` | calibra para onde você olha |
| `presence-guard preview` | janela com a câmera: quem é cada rosto, ângulos, reflexo e a decisão |
| `presence-guard status` | mostra se está tudo configurado |
| `presence-guard log` | últimas linhas do log do serviço |
| `presence-guard debug` | roda no terminal mostrando cada análise (Ctrl+C sai e religa o serviço) |
| `presence-guard esquecer` | apaga o cadastro do rosto |

Se o terminal disser que `presence-guard` não existe, saia e entre de novo na sessão
(o Ubuntu só põe `~/.local/bin` no PATH no login) ou use `~/.local/bin/presence-guard`.

### Atalho de teclado para ligar/desligar

Configurações → Teclado → Atalhos de teclado → Ver e personalizar → Atalhos personalizados → **+**

- Nome: `Presence Guard`
- Comando: `/home/SEU_USUARIO/.local/bin/presence-guard toggle`
- Atalho: por exemplo `Super+Shift+P`

## Configuração

Arquivo: `~/.config/presence-guard/config.ini` (todas as opções estão comentadas lá).
Depois de mudar: `systemctl --user restart presence-guard`.

| opção | padrão | |
|---|---|---|
| `borrar_apos` | 2.5 | segundos sem olhar até borrar |
| `bloquear_apos` | 60 | segundos sem você na frente até bloquear (fica borrado até lá) |
| `bloquear_estranho_apos` | 15 | outra pessoa no seu lugar bloqueia mais rápido |
| `olhos_fechados_apos` | 4 | olhos fechados contam como "não olhando" (0 desliga) |
| `margem_cabeca` | 8 | folga em graus; aumente se borrar enquanto você olha |
| `ignorar_olhos_com_oculos` | sim | de óculos usa só a cabeça (o IR reflete nas lentes) |
| `limite_dono` | 0.40 | semelhança mínima para ser você; abaixe se não te reconhecer |
| `outra_pessoa` | avisar | `desligado`, `avisar` ou `borrar` |
| `virou_o_rosto` | leve | nível do borrão ao virar o rosto: `leve`, `medio` ou `forte` |
| `outra_pessoa_olhando` | medio | nível quando outra pessoa olha junto |
| `escurecer_ate_bloquear` | sim | quando você sai, o borrão forte escurece aos poucos até bloquear |
| `considerar_teclado_e_mouse` | sim | digitando/mexendo o mouse conta como olhando |
| `pausar_borrao_em_videos` | sim | não borra com vídeo/chamada tocando (o bloqueio continua) |
| `desbloqueio_ao_chegar` | nao | com a tela bloqueada, chama o Howdy quando você volta |
| `analises_por_segundo` / `_alerta` | 1.5 / 4 | menos = gasta menos |

Clique na tela borrada para ignorar por 20 s, ou só mexa o mouse para tirar o borrão leve
(nenhum dos dois funciona quando é outra pessoa no seu lugar).

## Energia

- Ligado: a câmera IR fica ligada; o programa acorda 1,5x por segundo para analisar
  (4x quando algo muda). Na bateria, metade. Fora isso ele dorme: o Howdy, a calibração e a
  tela de bloqueio acordam ele na hora por eventos (nada de ficar conferindo toda hora).
- Medido numa máquina de teste mais lenta que o Core Ultra 5: ~3,5% de um núcleo vigiando,
  ~0% com a tela bloqueada. Memória: ~170 MB, ou ~260 MB com o reconhecimento de rosto.
- Com a tela bloqueada a câmera é desligada (a não ser que `desbloqueio_ao_chegar = sim`).
- Com a tampa da câmera fechada ele pausa sozinho (imagem escura não conta como "saiu").
- O reconhecimento (SFace) só é carregado se você cadastrar o rosto, e roda a cada 5 s.
- `presence-guard off`: gasto zero.

## Se algo não funcionar

Rode e me mande a saída de:

```bash
presence-guard status
presence-guard log
```

Problemas comuns:

- **Borra mesmo eu olhando**: rode `presence-guard calibrate` de novo; se continuar, aumente
  `margem_cabeca` (ex.: 12) ou use `usar_direcao_dos_olhos = nao`. O `presence-guard preview`
  mostra os números ao vivo.
- **Não me reconhece (borra como se fosse outra pessoa)**: `presence-guard cadastrar --adicionar`
  na situação em que falhou (de óculos, com outra luz); ou abaixe `limite_dono`. O preview
  mostra a semelhança de cada rosto.
- **sudo pede senha em vez do rosto**: confira se a linha existe com
  `grep -rn presence-guard-yield /etc/pam.d` (tem que estar logo acima da linha do Howdy).
- **"Câmera sem imagem"**: a tampa está fechada ou o emissor IR não acendeu
  (no Howdy isso costuma ser resolvido com o `linux-enable-ir-emitter`).
- **Não bloqueia**: veja o `presence-guard log`. Alguma extensão do GNOME que mexe na tela de
  bloqueio pode impedir o comando de bloqueio.
- **O borrão não aparece**: confirme que a sessão é X11 (`echo $XDG_SESSION_TYPE`).

## Desinstalar

```bash
./uninstall.sh
```

Remove o serviço, a linha do PAM e os arquivos (a sua config e o cadastro ficam em
`~/.config/presence-guard`).
