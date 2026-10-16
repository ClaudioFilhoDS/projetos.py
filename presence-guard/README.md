# Presence Guard

Usa a câmera infravermelha do ThinkPad (a mesma que o Howdy usa) para:

- **bloquear o Ubuntu** quando você sai da frente do notebook (padrão: 10 s sem ninguém);
- **borrar a tela** quando você está na frente mas não está olhando (padrão: 2,5 s);
- avisar (ou borrar) quando **outra pessoa** está olhando a sua tela;
- opcional: quando você volta, **chamar o Howdy sozinho** para desbloquear sem apertar nada.

Funciona no escuro (IR), roda tudo localmente com o MediaPipe e não salva nenhuma imagem.
O `sudo` com o Howdy continua funcionando: antes do Howdy rodar, o programa solta a câmera.

Feito para: Ubuntu 24.04, sessão **Ubuntu on Xorg** (X11), Howdy já funcionando.

## Instalar

```bash
sudo apt install -y git
git clone -b claude/upbeat-sagan-2jhkne https://github.com/ClaudioFilhoDS/projetos.py.git
cd projetos.py/presence-guard
./install.sh
```

O instalador:

1. instala `python3-venv python3-tk libnotify-bin xdotool` (pede seu rosto/senha);
2. cria um ambiente Python isolado em `~/.local/share/presence-guard` com MediaPipe;
3. coloca uma linha no PAM **antes** da linha do Howdy (`pam_exec ... presence-guard-yield`),
   com backup `*.bak-presence-guard`. Isso só avisa o programa para soltar a câmera; não
   muda a sua autenticação;
4. liga o serviço (inicia junto com a sua sessão);
5. oferece a calibração.

**Depois de instalar, teste o sudo em outro terminal** (deixe o do instalador aberto):

```bash
sudo -k; sudo echo funcionou
```

Se algo der errado, no terminal do instalador:
`for f in /etc/pam.d/*.bak-presence-guard; do sudo cp -a "$f" "${f%.bak-presence-guard}"; done`

## Calibrar (recomendado)

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
| `presence-guard calibrate` | calibra para onde você olha |
| `presence-guard preview` | janela com a imagem da câmera, ângulos e a decisão (OLHANDO / NAO OLHANDO) |
| `presence-guard status` | mostra se está tudo configurado |
| `presence-guard log` | últimas linhas do log do serviço |
| `presence-guard debug` | roda no terminal mostrando cada análise (Ctrl+C sai e religa o serviço) |

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
| `bloquear_apos` | 10 | segundos sem ninguém até bloquear |
| `olhos_fechados_apos` | 4 | olhos fechados contam como "não olhando" (0 desliga) |
| `margem_cabeca` | 8 | folga em graus; aumente se borrar enquanto você olha |
| `usar_direcao_dos_olhos` | sim | usa também para onde os olhos apontam |
| `outra_pessoa` | avisar | `desligado`, `avisar` ou `borrar` |
| `pausar_borrao_em_videos` | sim | não borra com vídeo/chamada tocando (o bloqueio continua) |
| `desbloqueio_ao_chegar` | nao | com a tela bloqueada, chama o Howdy quando você volta |
| `analises_por_segundo` / `_na_bateria` | 6 / 3 | menos = gasta menos |

Clique na tela borrada para ignorar por 20 s.

## Energia

- Ligado: a câmera IR fica ligada e o rosto é analisado 6x/s na tomada e 3x/s na bateria
  (uns poucos % de um núcleo do Core Ultra).
- Com a tela bloqueada a câmera é desligada (a não ser que `desbloqueio_ao_chegar = sim`).
- Com a tampa da câmera fechada ele pausa sozinho (imagem escura não conta como "saiu").
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

Remove o serviço, a linha do PAM e os arquivos (a sua config fica em `~/.config/presence-guard`).
