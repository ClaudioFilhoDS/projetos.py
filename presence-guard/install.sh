#!/usr/bin/env bash
# Instala o Presence Guard para o seu usuário (Ubuntu 24.04, sessão X11).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

APP_DIR="$HOME/.local/share/presence-guard"
CONF_DIR="$HOME/.config/presence-guard"
BIN="$HOME/.local/bin/presence-guard"
PAM_LINE=$'auth\toptional\tpam_exec.so quiet /usr/local/bin/presence-guard-yield'

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then
    echo "Rode sem sudo:  ./install.sh  (ele pede a senha só quando precisa)"; exit 1
fi
if [ "${XDG_SESSION_TYPE:-}" != "x11" ]; then
    warn "Sua sessão é '${XDG_SESSION_TYPE:-?}'. O borrão precisa da sessão 'Ubuntu on Xorg'."
fi

step "Instalando pacotes do sistema"
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-tk libnotify-bin xdotool x11-xserver-utils v4l-utils

step "Criando o ambiente Python (MediaPipe, OpenCV...)"
mkdir -p "$APP_DIR" "$CONF_DIR" "$HOME/.local/bin"
[ -x "$APP_DIR/venv/bin/python" ] || python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/venv/bin/pip" install --quiet mediapipe mss pillow

step "Copiando os arquivos"
install -m 755 presence_guard.py "$APP_DIR/presence_guard.py"
if [ -f "$CONF_DIR/config.ini" ]; then
    echo "config.ini já existe; mantive o seu."
else
    install -m 644 config.ini "$CONF_DIR/config.ini"
fi
cat > "$BIN" <<EOF
#!/bin/sh
exec "$APP_DIR/venv/bin/python" "$APP_DIR/presence_guard.py" "\$@"
EOF
chmod 755 "$BIN"
"$BIN" download-model

step "Integração com o Howdy (para o sudo continuar reconhecendo seu rosto)"
sudo install -m 755 presence-guard-yield /usr/local/bin/presence-guard-yield
mapfile -t PAM_FILES < <(grep -lE '^[[:space:]]*auth[[:space:]].*howdy' /etc/pam.d/* 2>/dev/null || true)
if [ "${#PAM_FILES[@]}" -eq 0 ]; then
    warn "Não achei o Howdy em /etc/pam.d; pulei. (Veja a seção PAM no README.)"
else
    for f in "${PAM_FILES[@]}"; do
        if grep -q presence-guard-yield "$f"; then
            echo "$f já está configurado."
            continue
        fi
        sudo cp -a "$f" "$f.bak-presence-guard"
        # coloca a linha logo ANTES da linha do Howdy (não mexe em mais nada)
        sudo python3 - "$f" "$PAM_LINE" <<'PY'
import re, sys
path, new = sys.argv[1], sys.argv[2] + "\n"
lines = open(path).read().splitlines(keepends=True)
for i, line in enumerate(lines):
    if re.match(r"\s*auth\s.*howdy", line):
        lines.insert(i, new)
        break
open(path, "w").write("".join(lines))
PY
        echo "Configurado: $f (backup em $f.bak-presence-guard)"
        grep -n -A1 presence-guard-yield "$f" | sed 's/^/    /'
    done
fi

step "Ligando o serviço (inicia junto com a sessão)"
mkdir -p "$HOME/.config/systemd/user"
install -m 644 presence-guard.service "$HOME/.config/systemd/user/presence-guard.service"
# garante que o serviço sabe qual é a sua tela (X11)
systemctl --user import-environment DISPLAY XAUTHORITY 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable presence-guard.service
systemctl --user restart presence-guard.service
sleep 2
systemctl --user --no-pager --lines=5 status presence-guard.service || true

printf '\n\033[1;32mInstalado!\033[0m\n'
cat <<EOF

Teste o sudo com o Howdy em OUTRO terminal (deixe este aberto):
    sudo -k; sudo echo funcionou
Se der problema, volte neste terminal e desfaça com:
    for f in /etc/pam.d/*.bak-presence-guard; do sudo cp -a "\$f" "\${f%.bak-presence-guard}"; done

EOF
read -r -p "Calibrar o olhar agora? (leva 20 segundos) [S/n] " answer
if [[ ! "$answer" =~ ^[nN] ]]; then
    "$BIN" calibrate || true
fi
cat <<'EOF'

Pronto! Comandos (se "presence-guard" não for encontrado, saia e entre de novo na sessão,
ou use ~/.local/bin/presence-guard):
    presence-guard off        desliga de verdade (zero CPU e câmera livre)
    presence-guard on         liga de novo
    presence-guard toggle     alterna (bom para um atalho de teclado)
    presence-guard calibrate  recalibra o olhar
    presence-guard preview    mostra o que a câmera está vendo
    presence-guard log        o que ele andou fazendo (mande isso se der problema)
EOF
