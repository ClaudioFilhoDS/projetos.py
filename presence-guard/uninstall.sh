#!/usr/bin/env bash
# Remove o Presence Guard (mantém o Howdy como estava antes).
set -uo pipefail

systemctl --user disable --now presence-guard.service 2>/dev/null
rm -f "$HOME/.config/systemd/user/presence-guard.service"
systemctl --user daemon-reload

# tira a linha do PAM dos arquivos de verdade (não dos backups)
grep -l presence-guard-yield /etc/pam.d/* 2>/dev/null \
    | grep -vE '\.(bak-presence-guard|dpkg-[a-z]+)$|~$' \
    | while read -r f; do
        sudo sed -i '/presence-guard-yield/d' "$f" && echo "Removido de $f"
    done
# os backups que o instalador fez não servem mais para nada
sudo rm -f /etc/pam.d/*.bak-presence-guard
sudo rm -f /usr/local/bin/presence-guard-yield /run/presence-guard.yield

rm -rf "$HOME/.local/share/presence-guard" "$HOME/.local/bin/presence-guard"
echo "Removido. Sua configuração e o cadastro do rosto ficaram em ~/.config/presence-guard"
echo "(apague essa pasta se quiser)."
