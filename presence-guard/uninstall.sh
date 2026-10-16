#!/usr/bin/env bash
# Remove o Presence Guard (mantém o Howdy como estava antes).
set -uo pipefail

systemctl --user disable --now presence-guard.service 2>/dev/null
rm -f "$HOME/.config/systemd/user/presence-guard.service"
systemctl --user daemon-reload

for f in $(grep -l presence-guard-yield /etc/pam.d/* 2>/dev/null); do
    sudo sed -i '/presence-guard-yield/d' "$f" && echo "Removido de $f"
done
sudo rm -f /usr/local/bin/presence-guard-yield /run/presence-guard.yield

rm -rf "$HOME/.local/share/presence-guard" "$HOME/.local/bin/presence-guard"
echo "Removido. Sua configuração ficou em ~/.config/presence-guard (apague se quiser)."
