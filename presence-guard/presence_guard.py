#!/usr/bin/env python3
"""Presence Guard: bloqueia o Ubuntu quando você sai da frente do notebook e
borra a tela quando você não está olhando para ela.

Usa a mesma câmera infravermelha do Howdy (funciona no escuro):
  - YuNet (OpenCV) acha os rostos na imagem, mesmo pequenos ou longe;
  - MediaPipe Face Landmarker analisa só o recorte do seu rosto (cabeça e olhos);
  - SFace (OpenCV) reconhece se é você, com ou sem óculos (presence-guard cadastrar).
Tudo roda localmente; nenhuma imagem é salva.

Comandos:
    on / off     liga / desliga de verdade (desligado = zero CPU e câmera livre)
    toggle       alterna entre ligado e desligado (bom para um atalho de teclado)
    cadastrar    grava o seu rosto com e sem óculos (para reconhecer você)
    calibrate    calibra para onde você olha quando está usando a tela
    preview      mostra a câmera com o que o programa está enxergando
    status       mostra se está tudo configurado
    log          mostra o que ele andou fazendo (para mandar se algo der errado)
    debug        roda no terminal mostrando cada análise (Ctrl+C para sair)
    esquecer     apaga o cadastro do seu rosto
    run          roda o vigia (é o que o serviço do systemd chama)
"""

import argparse
import concurrent.futures
import configparser
import ctypes
import ctypes.util
import glob
import io
import json
import logging
import math
import os
import queue
import re
import select
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # abre a câmera mais rápido
    try:  # pixels de verdade mesmo com a escala de 125%/150% do Windows
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)  # um vigia de fundo não precisa de todos os núcleos
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

APP = "presence-guard"
SERVICE = f"{APP}.service"
HOME = Path.home()
if IS_WINDOWS:
    CONFIG_DIR = Path(os.environ.get("APPDATA") or HOME / "AppData/Roaming") / APP
    DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or HOME / "AppData/Local") / APP
    RUNTIME_DIR = DATA_DIR / "run"
else:
    CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME") or HOME / ".config") / APP
    DATA_DIR = Path(os.environ.get("XDG_DATA_HOME") or HOME / ".local/share") / APP
    RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/{APP}-{os.getuid()}") / APP
# Windows: arquivo de log, "desligado" (vale até o próximo `on`), PIDs e câmeras achadas
LOG_FILE = DATA_DIR / "presence-guard.log"
DISABLED_FILE = CONFIG_DIR / "desligado"
PID_FILE = RUNTIME_DIR / "vigia.pid"
SUPERVISOR_PID = RUNTIME_DIR / "supervisor.pid"
CAMERAS_FILE = DATA_DIR / "cameras.json"
CONFIG_FILE = CONFIG_DIR / "config.ini"
CALIBRATION_FILE = CONFIG_DIR / "calibration.json"
FACES_FILE = CONFIG_DIR / "rostos.npz"
HOLD_FILE = RUNTIME_DIR / "hold"
# Existe enquanto alguém deste programa está com a câmera aberta (o cadastro, a
# calibração e o preview esperam o serviço soltar a câmera olhando para ele).
CAMERA_FLAG = RUNTIME_DIR / "camera-open"
# O PAM (presence-guard-yield) encosta neste arquivo logo antes do Howdy rodar. Fica na
# pasta do usuário porque no sudo o PAM roda o script como você, não como root.
YIELD_FILE = RUNTIME_DIR / "yield"
HOWDY_CONFIGS = (
    "/etc/howdy/config.ini",
    "/usr/local/etc/howdy/config.ini",
    "/lib/security/howdy/config.ini",
    "/usr/lib/security/howdy/config.ini",
    "/usr/lib64/security/howdy/config.ini",
)

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
_HF = "https://huggingface.co/opencv/"
MODELS = {  # nome: (arquivo, tamanho mínimo, endereços)
    "landmarker": ("face_landmarker.task", 3_000_000, [
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"]),
    "detector": ("face_detection_yunet_2023mar.onnx", 200_000, [
        _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        _HF + "face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"]),
    "recognizer": ("face_recognition_sface_2021dec.onnx", 30_000_000, [
        _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        _HF + "face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"]),
}

DETECT_WIDTH = 320         # o YuNet procura rostos numa cópia reduzida da imagem
CROP_SIZE = 256            # recorte do rosto entregue ao MediaPipe
MIN_FACE_WIDTH = 0.06      # rosto menor que isso (fração da imagem) é longe demais
OTHER_FACE_RATIO = 0.45    # outro rosto precisa ter pelo menos 45% do tamanho do seu
FACING_MAX = 0.45          # nariz fora do meio dos olhos além disso = rosto virado
BLINK_CLOSED = 0.55        # blendshape eyeBlink acima disso = olho fechado
GLARE_FRACTION = 0.04      # olho com mais que isso de pixels estourados = reflexo dos óculos
LOCKED_RATE = 1.0          # análises por segundo com a tela bloqueada
LEARN_WINDOW = 15.0        # segundos depois de desbloquear em que ele pode aprender seu rosto
MAX_AUTO_SAMPLES = 30
LABELS = {"sem_oculos": "sem óculos", "com_oculos": "com óculos"}
# Níveis do borrão: (quanto a imagem é reduzida antes de ampliar de volta, brilho)
BLUR_LEVELS = {
    1: (13, 0.90),   # leve: dá para ver as janelas, não dá para ler
    2: (18, 0.80),   # médio
    3: (32, 0.62),   # forte
    4: (32, 0.45),   # forte, escurecendo...
    5: (32, 0.30),   # ...até bloquear
}
LEVEL_NAMES = {"leve": 1, "medio": 2, "médio": 2, "forte": 3}
TURNED_MAX = 15.0          # por até 15 s, rosto sumido com a cabeça ainda ali = "virou o rosto"
HEAD_STILL_RATIO = 0.6     # a região do rosto continua com pelo menos 60% do brilho de antes
PAUSE_HOLD = "cadastro/calibração/preview usando a câmera"
PAUSE_HOWDY = "Howdy usando a câmera"
PAUSE_LOCKED = "tela bloqueada"

log = logging.getLogger(APP)


# --------------------------------------------------------------------------- config

DEFAULTS = {
    "camera": {"dispositivo": "auto"},
    "tempos": {
        "borrar_apos": "2.5",
        "bloquear_apos": "60",
        "bloquear_estranho_apos": "15",
        "olhos_fechados_apos": "4",
        "carencia_apos_desbloquear": "5",
    },
    "olhar": {
        "margem_cabeca": "8",
        "usar_direcao_dos_olhos": "sim",
        "margem_olhos": "0.15",
        "ignorar_olhos_com_oculos": "sim",
    },
    "reconhecimento": {
        "usar": "sim",
        "limite_dono": "0.40",
        "limite_estranho": "0.25",
        "aprender_sozinho": "sim",
    },
    "privacidade": {"outra_pessoa": "avisar"},
    "energia": {
        "analises_por_segundo": "1.5",
        "analises_por_segundo_alerta": "4",
        "economia_na_bateria": "sim",
    },
    "borrao": {
        "virou_o_rosto": "leve",
        "outra_pessoa_olhando": "medio",
        "escurecer_ate_bloquear": "sim",
    },
    "atividade": {
        "considerar_teclado_e_mouse": "sim",
        "segundos_de_atividade": "3",
    },
    "recursos": {
        "bloquear": "sim",
        "borrar": "sim",
        "pausar_borrao_em_videos": "sim",
        "desbloqueio_ao_chegar": "nao",
        "notificacoes": "sim",
    },
}


class Config:
    def __init__(self, path=CONFIG_FILE):
        cp = configparser.ConfigParser()
        cp.BOOLEAN_STATES = {**cp.BOOLEAN_STATES, "sim": True, "s": True,
                             "nao": False, "não": False, "n": False}
        cp.read_dict(DEFAULTS)
        cp.read(path, encoding="utf-8")
        num, flag = cp.getfloat, cp.getboolean
        self.device = cp.get("camera", "dispositivo").strip()
        self.blur_after = num("tempos", "borrar_apos")
        self.lock_after = num("tempos", "bloquear_apos")
        self.stranger_lock_after = num("tempos", "bloquear_estranho_apos")
        self.eyes_closed_after = num("tempos", "olhos_fechados_apos")
        self.grace = num("tempos", "carencia_apos_desbloquear")
        self.head_margin = num("olhar", "margem_cabeca")
        self.use_gaze = flag("olhar", "usar_direcao_dos_olhos")
        self.gaze_margin = num("olhar", "margem_olhos")
        self.ignore_eyes_with_glasses = flag("olhar", "ignorar_olhos_com_oculos")
        self.recognition = flag("reconhecimento", "usar")
        self.owner_threshold = num("reconhecimento", "limite_dono")
        self.stranger_threshold = num("reconhecimento", "limite_estranho")
        self.auto_learn = flag("reconhecimento", "aprender_sozinho")
        self.other_person = cp.get("privacidade", "outra_pessoa").strip().lower()
        self.rate = max(0.5, num("energia", "analises_por_segundo"))
        self.rate_alert = max(self.rate, num("energia", "analises_por_segundo_alerta"))
        self.battery_saver = flag("energia", "economia_na_bateria")
        self.lock = flag("recursos", "bloquear")
        self.blur = flag("recursos", "borrar")
        self.pause_blur_in_videos = flag("recursos", "pausar_borrao_em_videos")
        # no Windows o próprio Windows Hello já desbloqueia quando você chega
        self.wake_on_approach = flag("recursos", "desbloqueio_ao_chegar") and not IS_WINDOWS
        self.notifications = flag("recursos", "notificacoes")

        def level(key, default):
            return LEVEL_NAMES.get(cp.get("borrao", key).strip().lower(), default)

        self.level_look = level("virou_o_rosto", 1)
        self.level_other = level("outra_pessoa_olhando", 2)
        self.darken = flag("borrao", "escurecer_ate_bloquear")
        self.use_input = flag("atividade", "considerar_teclado_e_mouse")
        self.input_seconds = num("atividade", "segundos_de_atividade")


# --------------------------------------------------------------------------- sistema

def sh(*cmd, timeout=3):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def gdbus(dest, path, method, *args):
    r = sh("gdbus", "call", "--session", "--dest", dest, "--object-path", path,
           "--method", method, *args)
    return r.stdout.strip() if r and r.returncode == 0 else None


def screen_locked():
    return gdbus("org.gnome.ScreenSaver", "/org/gnome/ScreenSaver",
                 "org.gnome.ScreenSaver.GetActive") == "(true,)"


def lock_screen():
    if gdbus("org.gnome.ScreenSaver", "/org/gnome/ScreenSaver",
             "org.gnome.ScreenSaver.Lock") is not None:
        return True
    r = sh("xdg-screensaver", "lock")
    return bool(r and r.returncode == 0)


def idle_inhibited():
    """True quando algum app (vídeo, chamada) pediu para a tela não apagar."""
    return gdbus("org.gnome.SessionManager", "/org/gnome/SessionManager",
                 "org.gnome.SessionManager.IsInhibited", "8") == "(true,)"


def notify(title, body=""):
    sh("notify-send", "-a", "Presence Guard", "-i", "camera-web", title, body)


def on_battery():
    has_mains = False
    for supply in glob.glob("/sys/class/power_supply/*"):
        try:
            if Path(supply, "type").read_text().strip() != "Mains":
                continue
            has_mains = True
            if Path(supply, "online").read_text().strip() == "1":
                return False
        except OSError:
            continue
    return has_mains


def is_howdy_cmdline(argv):
    """O reconhecimento do Howdy (.../howdy/compare.py) ou o comando `howdy`."""
    if any("howdy" in arg and arg.endswith("/compare.py") for arg in argv[:3]):
        return True
    first = os.path.basename(argv[0])
    if first == "howdy":
        return True
    return (len(argv) > 1 and os.path.basename(argv[1]) == "howdy"
            and (first.startswith("python") or first in ("sudo", "pkexec")))


def howdy_running():
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/cmdline", "rb") as f:
                cmdline = f.read()
        except OSError:
            continue
        if b"howdy" in cmdline and is_howdy_cmdline(
                cmdline.decode(errors="replace").split("\0")):
            return True
    return False


def file_mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def process_alive(pid):
    return Path(f"/proc/{pid}").exists()


def terminate(pid):
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def pid_file_alive(path):
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return False
    return process_alive(pid)


@contextmanager
def camera_hold():
    """Pede para o serviço soltar a câmera enquanto cadastro/calibração/preview rodam."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    HOLD_FILE.write_text(str(os.getpid()))
    deadline = time.monotonic() + 3
    while pid_file_alive(CAMERA_FLAG) and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        yield
    finally:
        HOLD_FILE.unlink(missing_ok=True)


def laptop_monitor():
    """(x, y, largura, altura) da tela do notebook (eDP), ou da principal."""
    r = sh("xrandr", "--query")
    best = None
    if r and r.returncode == 0:
        for line in r.stdout.splitlines():
            m = re.match(r"^(\S+) connected (primary )?(\d+)x(\d+)\+(\d+)\+(\d+)", line)
            if not m:
                continue
            w, h, x, y = (int(v) for v in m.group(3, 4, 5, 6))
            if m.group(1).startswith("eDP"):
                return x, y, w, h
            if best is None or m.group(2):
                best = (x, y, w, h)
    if best:
        return best
    with screen_grabber() as sct:
        mon = sct.monitors[1]
    return mon["left"], mon["top"], mon["width"], mon["height"]


def screen_grabber():
    import mss
    return (getattr(mss, "MSS", None) or mss.mss)()  # mss >= 10 renomeou mss.mss


def model_path(name):
    return DATA_DIR / MODELS[name][0]


def ensure_models(*names):
    for name in names:
        filename, min_size, urls = MODELS[name]
        path = DATA_DIR / filename
        if path.exists() and path.stat().st_size >= min_size:
            continue
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        for url in urls:
            print(f"Baixando {filename}...", flush=True)
            try:
                urllib.request.urlretrieve(url, tmp)
            except OSError as e:
                print(f"  falhou ({e}); tentando outro endereço", flush=True)
                continue
            if tmp.stat().st_size >= min_size:
                tmp.replace(path)
                break
            print("  arquivo veio incompleto; tentando outro endereço", flush=True)
        else:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"não consegui baixar {filename}")


# --------------------------------------------------------------------------- Windows

if IS_WINDOWS:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _user32.OpenInputDesktop.restype = ctypes.c_void_p
    _user32.OpenInputDesktop.argtypes = [ctypes.c_uint, ctypes.c_bool, ctypes.c_uint]
    _user32.SwitchDesktop.argtypes = [ctypes.c_void_p]
    _user32.CloseDesktop.argtypes = [ctypes.c_void_p]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_bool, ctypes.c_uint]
    _kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    _kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    _kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint, ctypes.c_uint]
    _kernel32.GetTickCount.restype = ctypes.c_uint

    class _ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong),
                    ("th32ProcessID", ctypes.c_ulong), ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", ctypes.c_ulong), ("cntThreads", ctypes.c_ulong),
                    ("th32ParentProcessID", ctypes.c_ulong), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.c_ulong), ("szExeFile", ctypes.c_wchar * 260)]

    class _PowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    class _LastInput(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    _kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry)]
    _kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry)]

    # janelas do Windows Hello dentro da sessão (UAC, senha de app, passkey)
    HELLO_PROMPTS = {"credentialuibroker.exe", "consent.exe"}
    TOAST = None  # o vigia aponta isto para os avisos do Tk dele

    def screen_locked():  # noqa: F811
        """Tela bloqueada, UAC ou Ctrl+Alt+Del (área de trabalho segura): é a hora em que
        o Windows Hello precisa da câmera."""
        desk = _user32.OpenInputDesktop(0, False, 0x0100)  # DESKTOP_SWITCHDESKTOP
        if not desk:
            return True
        try:
            return not _user32.SwitchDesktop(desk)
        finally:
            _user32.CloseDesktop(desk)

    def lock_screen():  # noqa: F811
        return bool(_user32.LockWorkStation())

    def idle_inhibited():  # noqa: F811
        """Algum app (vídeo, chamada) pediu para a tela ficar ligada."""
        state = ctypes.c_ulong(0)
        try:
            res = ctypes.windll.powrprof.CallNtPowerInformation(
                16, None, 0, ctypes.byref(state), ctypes.sizeof(state))  # SystemExecutionState
        except (OSError, AttributeError):
            return False
        return res == 0 and bool(state.value & 0x2)  # ES_DISPLAY_REQUIRED

    def notify(title, body=""):  # noqa: F811
        if TOAST:
            TOAST(title, body)

    def on_battery():  # noqa: F811
        status = _PowerStatus()
        return bool(_kernel32.GetSystemPowerStatus(ctypes.byref(status))) and status.ACLineStatus == 0

    def process_names():
        snap = _kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if not snap or snap == ctypes.c_void_p(-1).value:
            return
        try:
            entry = _ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            ok = _kernel32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                yield entry.szExeFile.lower()
                ok = _kernel32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            _kernel32.CloseHandle(snap)

    def howdy_running():  # noqa: F811
        """No Windows: uma janela do Windows Hello aberta dentro da sessão."""
        return any(name in HELLO_PROMPTS for name in process_names())

    def process_alive(pid):  # noqa: F811
        handle = _kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong(0)
            ok = _kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            _kernel32.CloseHandle(handle)

    def terminate(pid):  # noqa: F811
        handle = _kernel32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
        if handle:
            _kernel32.TerminateProcess(handle, 0)
            _kernel32.CloseHandle(handle)

    def laptop_monitor():  # noqa: F811
        """Monitor principal (no Windows ele sempre começa em 0,0)."""
        return 0, 0, _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)


# --------------------------------------------------------------------------- câmera

class CameraError(Exception):
    pass


def howdy_device():
    for path in HOWDY_CONFIGS:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"^\s*device_path\s*=\s*(\S+)", text, re.M)
        if m and m.group(1).lower() != "none":
            return m.group(1)
    return None


def detect_ir_device():
    """Procura uma câmera que entregue imagem em tons de cinza (infravermelho)."""
    def number(path):
        return int(re.sub(r"\D", "", path) or 0)

    for dev in sorted(glob.glob("/dev/video*"), key=number):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                continue
            frame = None
            for _ in range(5):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is None:
                continue
            if frame.ndim == 2 or (np.array_equal(frame[..., 0], frame[..., 1])
                                   and np.array_equal(frame[..., 1], frame[..., 2])):
                return dev
        finally:
            cap.release()
    return None


def windows_cameras(max_index=4):
    """[(índice, é infravermelho?)] das câmeras que o Windows deixa abrir."""
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        try:
            if not cap.isOpened():
                continue
            frame = None
            for _ in range(5):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is None:
                continue
            gray = frame.ndim == 2 or (np.array_equal(frame[..., 0], frame[..., 1])
                                       and np.array_equal(frame[..., 1], frame[..., 2]))
            found.append([index, bool(gray)])
        finally:
            cap.release()
    return found


def resolve_device_windows(cfg, rescan=False):
    want = cfg.device.strip().lower()
    if want.isdigit():
        return int(want), "config.ini"
    if want in ("auto", "ir"):
        try:
            names = run_in_mta(windows_ir_sources)
        except Exception as e:
            names = []
            log.info("API de câmera do Windows indisponível (%s)", e)
        if names:
            return WINRT_IR, f"infravermelho: {names[0]}"
        log.warning("o Windows não mostra a câmera infravermelha para os apps; usando a normal")
    cams = None
    if not rescan:
        try:
            cams = json.loads(CAMERAS_FILE.read_text())
        except (OSError, ValueError):
            cams = None
    if not cams:
        cams = windows_cameras()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CAMERAS_FILE.write_text(json.dumps(cams))
    ir = [i for i, gray in cams if gray]
    rgb = [i for i, gray in cams if not gray]
    if want != "rgb" and ir:
        return ir[0], "infravermelho (OpenCV)"
    if rgb:
        return rgb[0], "câmera normal" + ("" if want == "rgb" else " (o infravermelho não abriu)")
    return 0, "padrão"


def resolve_device(cfg):
    if IS_WINDOWS:
        return resolve_device_windows(cfg)
    if cfg.device.lower() != "auto":
        return cfg.device, "config.ini"
    dev = howdy_device()
    if dev:
        return dev, "mesma do Howdy"
    dev = detect_ir_device()
    if dev:
        return dev, "infravermelho detectado"
    return "/dev/video0", "padrão"


class Camera:
    """Fica aberta, mas só tira quadros quando precisa (o resto do tempo dorme)."""

    def __init__(self, device):
        self.device = device
        self.cap = None

    @property
    def is_open(self):
        return self.cap is not None

    def open(self):
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        CAMERA_FLAG.write_text(str(os.getpid()))  # antes de abrir: ninguém nos pega no meio
        if IS_WINDOWS:
            cap = cv2.VideoCapture(int(self.device), cv2.CAP_MSMF)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(int(self.device), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(os.path.realpath(self.device), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            CAMERA_FLAG.unlink(missing_ok=True)
            raise CameraError(f"não consegui abrir a câmera {self.device} (em uso ou inexistente)")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        # os primeiros quadros saem escuros enquanto a exposição/emissor IR acordam
        if sum(cap.grab() for _ in range(5)) == 0:
            cap.release()
            CAMERA_FLAG.unlink(missing_ok=True)
            raise CameraError(f"a câmera {self.device} não entregou imagem (em uso?)")
        self.cap = cap

    def fresh_frame(self, pick=2):
        """Um quadro de agora (não um que ficou parado no buffer), o mais claro de
        `pick` seguidos: câmeras IR costumam alternar quadros com o emissor apagado."""
        for _ in range(8):
            started = time.monotonic()
            if not self.cap.grab():
                return None
            if time.monotonic() - started > 0.012:  # precisou esperar = quadro novo
                break
        best, best_mean = None, -1.0
        for i in range(pick):
            if i and not self.cap.grab():
                break
            ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            mean = float(frame[::4, ::4].mean())
            if mean > best_mean:
                best, best_mean = frame, mean
        return best

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            CAMERA_FLAG.unlink(missing_ok=True)


WINRT_IR = "winrt-ir"  # "dispositivo" que quer dizer: a câmera IR do Windows Hello, via WinRT


def winrt_init():
    """A API de câmera do Windows (WinRT) quer a thread num apartamento COM multithread."""
    try:
        from winrt.runtime import ApartmentType, init_apartment

        init_apartment(ApartmentType.MULTI_THREADED)
    except Exception:  # já iniciado nesta thread, ou pywinrt antigo: segue assim mesmo
        pass


def run_in_mta(func, *args):
    """Roda `func` numa thread própria (MTA), sem mexer no apartamento da thread do Tk."""
    with concurrent.futures.ThreadPoolExecutor(1, initializer=winrt_init) as pool:
        return pool.submit(func, *args).result()


def windows_ir_sources():
    """Nomes das câmeras infravermelhas que o Windows mostra para os apps."""
    from winrt.windows.media.capture.frames import MediaFrameSourceGroup, MediaFrameSourceKind

    names = []
    for group in MediaFrameSourceGroup.find_all_async().get():
        if any(info.source_kind == MediaFrameSourceKind.INFRARED for info in group.source_infos):
            names.append(group.display_name)
    return names


def bitmap_to_gray(bitmap):
    """SoftwareBitmap (Gray8, Gray16, NV12 ou outro) -> imagem cinza 8 bits do numpy."""
    from winrt.windows.graphics.imaging import (BitmapBufferAccessMode, BitmapPixelFormat,
                                                 SoftwareBitmap)

    if bitmap is None:
        return None
    fmt = bitmap.bitmap_pixel_format
    if fmt not in (BitmapPixelFormat.GRAY8, BitmapPixelFormat.GRAY16, BitmapPixelFormat.NV12):
        bitmap = SoftwareBitmap.convert(bitmap, BitmapPixelFormat.GRAY8)
        fmt = BitmapPixelFormat.GRAY8
    with bitmap.lock_buffer(BitmapBufferAccessMode.READ) as buffer:
        plane = buffer.get_plane_description(0)  # Gray8/Gray16: a imagem; NV12: o Y (cinza)
        with buffer.create_reference() as ref:
            raw = np.frombuffer(memoryview(ref), dtype=np.uint8)
            rows = raw[plane.start_index:plane.start_index + plane.stride * plane.height]
            rows = rows.reshape(plane.height, plane.stride).copy()
    if fmt == BitmapPixelFormat.GRAY16:
        img = rows.view(np.uint16)[:, :plane.width]
        shift = max(0, int(img.max()).bit_length() - 8)  # sensores IR costumam usar 10-12 bits
        return (img >> shift).astype(np.uint8)
    return rows[:, :plane.width]


class WindowsIRCamera:
    """A câmera infravermelha do Windows Hello, lida pela API de câmera do próprio Windows
    (o OpenCV não enxerga ela). Tenta o controle exclusivo, que deixa acender o emissor IR;
    se o Windows não deixar, lê compartilhando com quem já está usando. Todas as chamadas
    WinRT rodam numa thread própria (MTA), venha a chamada de onde vier."""

    def __init__(self, device=WINRT_IR):
        self.device = device
        self.pool = None
        self.capture = self.reader = None
        self.exclusive = False
        self.torch = "?"
        self.last_time = None

    @property
    def is_open(self):
        return self.reader is not None

    def _call(self, func, *args):
        if self.pool is None:
            self.pool = concurrent.futures.ThreadPoolExecutor(
                1, thread_name_prefix="winrt", initializer=winrt_init)
        return self.pool.submit(func, *args).result()

    def open(self):
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        CAMERA_FLAG.write_text(str(os.getpid()))
        try:
            self._call(self._open)
        except Exception as e:
            self.close()
            raise CameraError(f"a câmera infravermelha do Windows não abriu: {e}") from e

    def _open(self):
        from winrt.windows.media.capture import (MediaCapture, MediaCaptureInitializationSettings,
                                                  MediaCaptureMemoryPreference,
                                                  MediaCaptureSharingMode, StreamingCaptureMode)
        from winrt.windows.media.capture.frames import (MediaFrameReaderAcquisitionMode,
                                                         MediaFrameReaderStartStatus,
                                                         MediaFrameSourceGroup,
                                                         MediaFrameSourceKind)

        group = info = None
        for g in MediaFrameSourceGroup.find_all_async().get():
            info = next((i for i in g.source_infos
                         if i.source_kind == MediaFrameSourceKind.INFRARED), None)
            if info is not None:
                group = g
                break
        if group is None:
            raise CameraError("o Windows não mostra nenhuma câmera infravermelha para os apps")
        errors = []
        for mode in (MediaCaptureSharingMode.EXCLUSIVE_CONTROL,
                     MediaCaptureSharingMode.SHARED_READ_ONLY):
            settings = MediaCaptureInitializationSettings()
            settings.source_group = group
            settings.sharing_mode = mode
            settings.memory_preference = MediaCaptureMemoryPreference.CPU
            settings.streaming_capture_mode = StreamingCaptureMode.VIDEO
            capture = MediaCapture()
            try:
                capture.initialize_with_settings_async(settings).get()
            except Exception as e:
                errors.append(f"{mode.name}: {e}")
                capture.close()
                continue
            self.capture = capture
            self.exclusive = mode == MediaCaptureSharingMode.EXCLUSIVE_CONTROL
            break
        if self.capture is None:
            raise CameraError("; ".join(errors))
        if self.exclusive:
            self._torch_on()
        reader = self.capture.create_frame_reader_async(self.capture.frame_sources[info.id]).get()
        reader.acquisition_mode = MediaFrameReaderAcquisitionMode.REALTIME
        status = reader.start_async().get()
        if status != MediaFrameReaderStartStatus.SUCCESS:
            reader.close()
            raise CameraError(f"a leitura não começou ({status.name})")
        self.reader = reader
        if self._fresh_frame(3.0) is None:
            raise CameraError("a câmera infravermelha não mandou imagem")

    def _torch_on(self):
        """Acende o emissor IR (alternado = metade do gasto; os quadros apagados são pulados)."""
        from winrt.windows.media.devices import InfraredTorchMode

        try:
            torch = self.capture.video_device_controller.infrared_torch_control
            if not torch.is_supported:
                self.torch = "o driver não deixa controlar (fica por conta dele)"
                return
            modes = list(torch.supported_modes)
            mode = (InfraredTorchMode.ALTERNATING_FRAME_ILLUMINATION
                    if InfraredTorchMode.ALTERNATING_FRAME_ILLUMINATION in modes
                    else InfraredTorchMode.ON)
            torch.current_mode = mode
            self.torch = "alternado" if mode == InfraredTorchMode.ALTERNATING_FRAME_ILLUMINATION else "ligado"
        except Exception as e:
            self.torch = f"sem controle ({e})"

    def fresh_frame(self, pick=2):
        if self.reader is None:
            return None
        try:
            return self._call(self._fresh_frame, 0.8)
        except Exception as e:
            log.warning("erro lendo a câmera infravermelha: %s", e)
            return None

    def _fresh_frame(self, timeout):
        """O próximo quadro novo iluminado pelo emissor IR."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ref = self.reader.try_acquire_latest_frame()
            if ref is None:
                time.sleep(0.02)
                continue
            gray = None
            try:
                stamp = ref.system_relative_time
                if stamp is not None and stamp == self.last_time:
                    time.sleep(0.02)
                    continue
                self.last_time = stamp
                video = ref.video_media_frame
                try:
                    infrared = video.infrared_media_frame if video is not None else None
                except Exception:
                    infrared = None
                if infrared is not None and not infrared.is_illuminated:
                    continue  # quadro com o emissor apagado: escuro, pula
                gray = bitmap_to_gray(video.software_bitmap) if video is not None else None
            finally:
                ref.close()
            if gray is not None:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return None

    def close(self):
        if self.pool is not None:
            try:
                self._call(self._close)
            except Exception:
                pass
            self.pool.shutdown(wait=False)
            self.pool = None
        CAMERA_FLAG.unlink(missing_ok=True)

    def _close(self):
        reader, capture = self.reader, self.capture
        self.reader = self.capture = None
        if reader is not None:
            try:
                reader.stop_async().get()
            except Exception:
                pass
            reader.close()
        if capture is not None:
            capture.close()  # solta a câmera para o Windows Hello


def make_camera(device):
    return WindowsIRCamera(device) if device == WINRT_IR else Camera(device)


def is_dark(frame):
    """Tampa da câmera fechada, emissor IR desligado ou quarto escuro (câmera RGB)."""
    small = frame[::8, ::8]
    return float(small.mean()) < 12 and float(small.std()) < 10


# --------------------------------------------------------------------------- rosto

@dataclass
class Pose:
    yaw: float       # graus; ~0 = virado para a câmera
    pitch: float
    gaze_h: float    # direção dos olhos (blendshapes, -1..1)
    gaze_v: float
    blink: float     # 0 = olhos abertos, 1 = fechados
    glare: bool      # reflexo forte nos olhos (óculos no infravermelho)
    points: list = field(default_factory=list, repr=False)  # no recorte, 0..1


def head_angles(matrix):
    r = np.asarray(matrix)[:3, :3]
    fx, fy, fz = r[:, 2]   # para onde o nariz aponta, no espaço da câmera
    yaw = math.degrees(math.atan2(fx, fz))
    pitch = math.degrees(math.atan2(-fy, math.hypot(fx, fz)))
    return yaw, pitch


def facing_ratio(row):
    """Quanto o nariz está fora do meio dos olhos (pontos do YuNet); 0 = de frente."""
    rex, rey, lex, ley, nx, _ny = row[4:10]
    dist = math.hypot(lex - rex, ley - rey) + 1e-6
    return (nx - (rex + lex) / 2) / dist


LEFT_EYE = (33, 133, 159, 145, 160, 144, 158, 153)
RIGHT_EYE = (362, 263, 386, 374, 387, 373, 385, 380)


def eye_glare(gray, landmarks):
    h, w = gray.shape
    for idx in (LEFT_EYE, RIGHT_EYE):
        xs = [landmarks[i].x * w for i in idx]
        ys = [landmarks[i].y * h for i in idx]
        pad_x, pad_y = (max(xs) - min(xs)) * 0.4, (max(ys) - min(ys)) * 0.8 + 2
        x0, x1 = int(max(0, min(xs) - pad_x)), int(min(w, max(xs) + pad_x))
        y0, y1 = int(max(0, min(ys) - pad_y)), int(min(h, max(ys) + pad_y))
        patch = gray[y0:y1, x0:x1]
        if patch.size and float((patch >= 245).mean()) > GLARE_FRACTION:
            return True
    return False


class FaceEngine:
    def __init__(self, landmarker=True, recognizer=False):
        ensure_models("detector")
        self.detector = cv2.FaceDetectorYN.create(str(model_path("detector")), "",
                                                  (DETECT_WIDTH, 180), 0.7, 0.3, 5)
        self.det_size = (DETECT_WIDTH, 180)
        self.landmarker = None
        self.recognizer = None
        if landmarker:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            ensure_models("landmarker")
            self._mp = mp
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path("landmarker"))),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        if recognizer:
            self.load_recognizer()

    def load_recognizer(self):
        if self.recognizer is None:
            ensure_models("recognizer")
            self.recognizer = cv2.FaceRecognizerSF.create(str(model_path("recognizer")), "")

    def detect(self, frame):
        """Rostos (linhas do YuNet em pixels da imagem), do maior para o menor."""
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / w
        size = (DETECT_WIDTH, max(1, round(h * scale)))
        if size != self.det_size:
            self.detector.setInputSize(size)
            self.det_size = size
        _, faces = self.detector.detect(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        if faces is None:
            return []
        faces = faces.copy()
        faces[:, :14] /= scale
        return sorted(faces, key=lambda r: r[2] * r[3], reverse=True)

    def pose(self, frame, row):
        """Cabeça e olhos do rosto `row`, analisando só um recorte ao redor dele."""
        x, y, w, h = row[:4]
        cx, cy, side = x + w / 2, y + h / 2, max(w, h) * 1.9
        k = CROP_SIZE / side
        m = np.float32([[k, 0, CROP_SIZE / 2 - cx * k], [0, k, CROP_SIZE / 2 - cy * k]])
        crop = cv2.warpAffine(frame, m, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                       data=rgb))
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        yaw, pitch = head_angles(result.facial_transformation_matrixes[0])
        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return Pose(
            yaw=yaw,
            pitch=pitch,
            gaze_h=((bs["eyeLookOutLeft"] - bs["eyeLookInLeft"])
                    + (bs["eyeLookInRight"] - bs["eyeLookOutRight"])) / 2,
            gaze_v=((bs["eyeLookUpLeft"] + bs["eyeLookUpRight"])
                    - (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"])) / 2,
            blink=(bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2,
            glare=eye_glare(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), landmarks),
            points=[(p.x, p.y) for p in landmarks],
        )

    def embedding(self, frame, row):
        """"Impressão digital" do rosto (vetor de 128 números, tamanho 1)."""
        feature = self.recognizer.feature(self.recognizer.alignCrop(frame, row)).flatten()
        return feature / (np.linalg.norm(feature) + 1e-9)

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()


class Gallery:
    """Os rostos cadastrados (com e sem óculos) + o que ele aprendeu sozinho."""

    def __init__(self, path=FACES_FILE, empty=False):
        self.path = path
        self.emb = np.zeros((0, 128), np.float32)
        self.labels = np.zeros(0, "<U12")
        self.auto = np.zeros(0, bool)
        if not empty and path.exists():
            try:
                with np.load(path) as data:
                    self.emb = data["emb"].astype(np.float32)
                    self.labels = data["labels"].astype("<U12")
                    self.auto = data["auto"].astype(bool)
            except (OSError, ValueError, KeyError):
                log.warning("cadastro de rostos inválido; ignorando (rode: presence-guard cadastrar)")

    @property
    def enrolled(self):
        return bool((~self.auto).any())

    def match(self, embedding):
        """(semelhança 0..1, rótulo da amostra mais parecida)"""
        if not len(self.emb):
            return 0.0, ""
        sims = self.emb @ embedding
        i = int(np.argmax(sims))
        return float(sims[i]), str(self.labels[i])

    def add(self, embedding, label, auto=False):
        self.emb = np.vstack([self.emb, embedding[None].astype(np.float32)])
        self.labels = np.append(self.labels, label)
        self.auto = np.append(self.auto, auto)
        if auto and self.auto.sum() > MAX_AUTO_SAMPLES:
            oldest = int(np.flatnonzero(self.auto)[0])
            keep = np.arange(len(self.auto)) != oldest
            self.emb, self.labels, self.auto = self.emb[keep], self.labels[keep], self.auto[keep]

    def counts(self):
        return {label: int(((self.labels == label) & ~self.auto).sum()) for label in LABELS}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez(buf, emb=self.emb, labels=self.labels, auto=self.auto)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(self.path)


@dataclass
class Track:
    """Um rosto acompanhado entre uma análise e outra, com quem ele é."""
    box: tuple
    seen: float
    status: str = "?"    # "dono", "estranho" ou "?" (ainda não sei)
    sim: float = 0.0
    label: str = ""
    low_hits: int = 0
    checked: float = -1e9


class Tracker:
    def __init__(self):
        self.tracks = []

    def update(self, rows, now):
        out, free = [], list(self.tracks)
        for row in rows:
            x, y, w, h = (float(v) for v in row[:4])
            cx, cy = x + w / 2, y + h / 2
            best, best_d = None, None
            for t in free:
                tx, ty, tw, th = t.box
                d = math.hypot(cx - (tx + tw / 2), cy - (ty + th / 2))
                if d < 0.6 * max(w, tw) and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is None:
                best = Track(box=(x, y, w, h), seen=now)
            else:
                free.remove(best)
                if now - best.seen > 0.7:
                    best.checked = -1e9  # sumiu e voltou: pode ser outra pessoa, confere já
            best.box, best.seen = (x, y, w, h), now
            out.append(best)
        # quem sumiu por um instante (detecção piscou) continua lembrado por 1,5 s
        self.tracks = out + [t for t in free if now - t.seen < 1.5]
        return out

    def clear(self):
        self.tracks = []


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


class Attention:
    """Decide se você está olhando para a tela, a partir da calibração."""

    # Sem calibração: cabeça mais ou menos virada para o notebook.
    UNCALIBRATED = {"center": {"yaw": 0.0, "pitch": 0.0},
                    "ranges": {"yaw": [-20.0, 20.0], "pitch": [-20.0, 20.0]}}

    def __init__(self, cfg):
        self.cfg = cfg
        data = self.UNCALIBRATED
        self.calibrated = False
        if CALIBRATION_FILE.exists():
            try:
                data = json.loads(CALIBRATION_FILE.read_text())
                self.calibrated = True
            except (OSError, ValueError):
                log.warning("calibration.json inválido; usando valores padrão")
        self.center = data["center"]
        self.ranges = data["ranges"]

    def features(self, pose):
        return {
            "yaw": wrap(pose.yaw - self.center["yaw"]),
            "pitch": wrap(pose.pitch - self.center["pitch"]),
            "gaze_h": pose.gaze_h,
            "gaze_v": pose.gaze_v,
        }

    def check(self, pose, use_eyes=True):
        """(olhando?, motivo)"""
        feats = self.features(pose)
        for key, (lo, hi) in self.ranges.items():
            if key.startswith("gaze"):
                if not use_eyes:
                    continue
                margin = self.cfg.gaze_margin
            else:
                margin = self.cfg.head_margin
            if not lo - margin <= feats[key] <= hi + margin:
                return False, key
        return True, ""


def x_init_threads():
    """Liga o modo multi-thread do Xlib (o Tk usa uma conexão, a thread da câmera outra)."""
    try:
        ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6").XInitThreads()
    except (OSError, AttributeError):
        pass


class _XssInfo(ctypes.Structure):
    _fields_ = [("window", ctypes.c_ulong), ("state", ctypes.c_int), ("kind", ctypes.c_int),
                ("til_or_since", ctypes.c_ulong), ("idle", ctypes.c_ulong),
                ("event_mask", ctypes.c_ulong)]


class InputIdle:
    """Há quantos segundos você não mexe no teclado/mouse (X11, extensão XScreenSaver)."""

    def __init__(self):
        self.ok = False
        try:
            x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
            xss = ctypes.CDLL(ctypes.util.find_library("Xss") or "libXss.so.1")
        except OSError:
            log.info("libXss indisponível: teclado/mouse não serão considerados")
            return
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        xss.XScreenSaverAllocInfo.restype = ctypes.POINTER(_XssInfo)
        xss.XScreenSaverQueryInfo.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                              ctypes.POINTER(_XssInfo)]
        self.display = x11.XOpenDisplay(None)
        if not self.display:
            return
        self.root = x11.XDefaultRootWindow(self.display)
        self.info = xss.XScreenSaverAllocInfo()
        self.xss = xss
        self.ok = bool(self.info)

    def seconds(self):
        if not self.ok or not self.xss.XScreenSaverQueryInfo(self.display, self.root, self.info):
            return None
        return self.info.contents.idle / 1000.0


if IS_WINDOWS:
    class InputIdle:  # noqa: F811
        """No Windows a conta é direta: GetLastInputInfo."""
        ok = True

        def seconds(self):
            info = _LastInput(ctypes.sizeof(_LastInput), 0)
            if not _user32.GetLastInputInfo(ctypes.byref(info)):
                return None
            return ((_kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


def region_luma(frame, box):
    """Brilho médio da imagem dentro de `box` (x, y, w, h)."""
    x, y, w, h = (int(round(v)) for v in box)
    patch = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
    return float(patch.mean()) if patch.size else 0.0


class Waker:
    """Faz a thread da câmera dormir até a próxima análise, mas acordar na hora se
    algo importante acontecer: Howdy pedindo a câmera (inotify em /run), cadastro ou
    calibração começando (inotify na pasta de runtime), tela bloqueando (wake())."""

    IN_ATTRIB, IN_CLOSE_WRITE, IN_MOVED_TO, IN_CREATE, IN_DELETE = 0x4, 0x8, 0x80, 0x100, 0x200

    def __init__(self):
        self.inotify = -1
        if IS_WINDOWS:  # sem inotify nem select em pipes: um Event e confere 1x por segundo
            self.event = threading.Event()
            return
        self.read_fd, self.write_fd = os.pipe()
        for fd in (self.read_fd, self.write_fd):
            os.set_blocking(fd, False)
        self.names = {YIELD_FILE.name, HOLD_FILE.name}
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if fd >= 0:
                RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                mask = (self.IN_ATTRIB | self.IN_CLOSE_WRITE | self.IN_MOVED_TO
                        | self.IN_CREATE | self.IN_DELETE)
                dirs = {str(YIELD_FILE.parent), str(RUNTIME_DIR)}
                if all(libc.inotify_add_watch(fd, d.encode(), mask) >= 0 for d in dirs):
                    self.inotify = fd
                else:
                    os.close(fd)
        except (OSError, AttributeError):
            pass
        if self.inotify < 0:
            log.info("inotify indisponível: conferindo os arquivos 2x por segundo")

    def wake(self):
        """Pode ser chamado de qualquer thread."""
        if IS_WINDOWS:
            self.event.set()
            return
        try:
            os.write(self.write_fd, b"!")
        except BlockingIOError:
            pass

    def wait(self, timeout):
        """Dorme até `timeout` segundos ou até algo importante acontecer."""
        if IS_WINDOWS:
            self.event.wait(max(0.0, min(timeout, 1.0)))
            self.event.clear()
            return
        if self.inotify < 0:
            timeout = min(timeout, 0.5)
        deadline = time.monotonic() + max(0.0, timeout)
        fds = [self.read_fd] + ([self.inotify] if self.inotify >= 0 else [])
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return
            ready, _, _ = select.select(fds, [], [], left)
            if not ready:
                return
            if self.read_fd in ready:
                try:
                    os.read(self.read_fd, 4096)
                except BlockingIOError:
                    pass
                return
            if self._relevant():
                return

    def _relevant(self):
        try:
            buf = os.read(self.inotify, 65536)
        except BlockingIOError:
            return False
        i, found = 0, False
        while i + 16 <= len(buf):
            _wd, _mask, _cookie, size = struct.unpack_from("iIII", buf, i)
            name = buf[i + 16:i + 16 + size].split(b"\0", 1)[0].decode(errors="replace")
            found = found or name in self.names
            i += 16 + size
        return found


class LockWatcher:
    """Acompanha a tela de bloqueio do GNOME pelos sinais do D-Bus, sem ficar perguntando."""

    def __init__(self, on_change=lambda: None):
        self.on_change = on_change
        self.locked = screen_locked()
        self.proc = None
        self.stop = threading.Event()
        threading.Thread(target=self._run, name="lock-watch", daemon=True).start()

    def _set(self, locked):
        if locked != self.locked:
            self.locked = locked
            self.on_change()

    def _run(self):
        if IS_WINDOWS:
            while not self.stop.wait(0.5):
                self._set(screen_locked())
            return
        while not self.stop.is_set():
            try:
                self.proc = subprocess.Popen(
                    ["gdbus", "monitor", "--session", "--dest", "org.gnome.ScreenSaver",
                     "--object-path", "/org/gnome/ScreenSaver"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except OSError:
                while not self.stop.wait(2):  # sem gdbus: pergunta a cada 2 s
                    self._set(screen_locked())
                return
            self.resync()
            for line in self.proc.stdout:
                if "ActiveChanged" in line:
                    self._set("true" in line)
            self.proc.wait()
            self.stop.wait(2)

    def resync(self):
        self._set(screen_locked())

    def close(self):
        self.stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


# --------------------------------------------------------------------------- vigia

class Guard:
    """Thread que olha a câmera de vez em quando e decide quando borrar/bloquear."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.ui = None
        self.cam = make_camera(device)
        self.gallery = Gallery()
        self.engine = FaceEngine(recognizer=self.wants_recognition())
        self.tracker = Tracker()
        self.attention = Attention(cfg)
        self.watcher = None
        self.waker = Waker()
        self.file_mtimes = self.config_mtimes()
        self.stop = threading.Event()
        self.crashed = False
        now = time.monotonic()
        self.locked = False
        self.battery = False
        self.blind = False
        self.alert = False
        self.blur_reason = None
        self.blur_level = 0
        self.input = None
        self.face_box = None
        self.face_luma = None
        self.snooze_until = 0.0
        self.grace_until = now + cfg.grace
        self.lock_requested = -1e9
        self.unlocked_at = -1e9
        self.learn_pending = False
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        if not YIELD_FILE.exists():
            YIELD_FILE.write_bytes(b"")
        self.yield_seen = file_mtime(YIELD_FILE)
        self.yield_until = 0.0
        self.inhibit_checked = -1e9
        self.inhibited = False
        self.other_notified = -1e9
        self.wake_attempts = 0
        self.wake_cooldown = 0.0
        self.absent_since = None
        self.face_since = None
        self.next = {"howdy": 0.0, "battery": 0.0, "files": 0.0, "resync": now + 30}
        self.reset_presence(now)

    # -- estado

    def wants_recognition(self):
        return self.cfg.recognition and self.gallery.enrolled

    def recognition_active(self):
        return self.wants_recognition() and self.engine.recognizer is not None

    @staticmethod
    def config_mtimes():
        return file_mtime(CALIBRATION_FILE), file_mtime(FACES_FILE)

    def reset_presence(self, now):
        self.last_present = self.last_looking = now
        self.looking_streak = 0
        self.eyes_closed_since = None
        self.dark_since = None
        self.other_since = None
        self.stranger_since = None

    def sleep(self, seconds):
        """Dorme até `seconds`, acordando antes se o Howdy pedir a câmera, a tela bloquear etc."""
        if not self.stop.is_set():
            self.waker.wait(seconds)

    def shutdown(self):
        self.stop.set()
        self.waker.wake()

    def snooze(self, seconds=20):
        """Clique na tela borrada. Não vale quando quem está na frente é outra pessoa."""
        if self.blur_reason == "estranho":
            return False
        self.snooze_until = time.monotonic() + seconds
        return True

    def set_blur(self, reason, level=3):
        level = level if reason else 0
        self.blur_reason = reason
        if level != self.blur_level:
            self.blur_level = level
            if self.ui:
                self.ui.request(level)

    def darkening(self, elapsed, limit):
        """Borrão forte que vai escurecendo até a hora de bloquear (contagem sem números)."""
        if not self.cfg.darken:
            return 3
        span = max(1.0, limit - self.cfg.blur_after)
        return 3 + min(2, max(0, int(3 * (elapsed - self.cfg.blur_after) / span)))

    def rate(self):
        if self.locked:
            rate = LOCKED_RATE
        else:
            rate = self.cfg.rate_alert if self.alert else self.cfg.rate
        if self.battery and self.cfg.battery_saver:
            rate = max(0.5, rate / 2)
        return rate

    def video_playing(self, now):
        if now - self.inhibit_checked >= 2:
            self.inhibit_checked = now
            self.inhibited = idle_inhibited()
        return self.inhibited

    def pause_reason(self, now):
        if pid_file_alive(HOLD_FILE):
            return PAUSE_HOLD
        if now < self.yield_until:
            return PAUSE_HOWDY
        if self.locked and not self.cfg.wake_on_approach:
            return PAUSE_LOCKED
        return None

    def pause_timeout(self, reason, now):
        """Quanto dormir pausado: os eventos acordam antes, isto é só a rede de segurança."""
        if reason == PAUSE_HOWDY:
            return max(0.1, self.yield_until - now)
        if reason == PAUSE_LOCKED:
            return 30.0
        return 2.0  # cadastro/calibração: confere se o processo ainda existe

    def housekeeping(self, now):
        locked = self.watcher.locked
        if locked != self.locked:
            self.locked = locked
            self.tracker.clear()
            if locked:
                log.info("tela bloqueada")
                self.set_blur(None)
                self.wake_attempts = 0
                self.face_since = self.absent_since = None
            else:
                log.info("tela desbloqueada")
                self.grace_until = now + self.cfg.grace
                self.snooze_until = 0.0
                self.unlocked_at = now
                self.learn_pending = self.cfg.auto_learn
                self.reset_presence(now)
        mtime = file_mtime(YIELD_FILE)
        if mtime > self.yield_seen:
            self.yield_seen = mtime
            self.yield_until = now + 4.0
        if now >= self.next["howdy"]:
            self.next["howdy"] = now + 2
            if howdy_running():
                self.yield_until = max(self.yield_until, now + 1.5)
        if now >= self.next["battery"]:
            self.next["battery"] = now + 30
            self.battery = on_battery()
        if now >= self.next["resync"]:
            self.next["resync"] = now + 30
            self.watcher.resync()
        if now >= self.next["files"]:
            self.next["files"] = now + 5
            mtimes = self.config_mtimes()
            if mtimes != self.file_mtimes:
                self.file_mtimes = mtimes
                self.attention = Attention(self.cfg)
                self.gallery = Gallery()
                if self.wants_recognition():
                    self.engine.load_recognizer()
                self.tracker.clear()
                log.info("calibração/cadastro recarregados")

    # -- laço principal

    def run(self):
        try:
            self.watcher = LockWatcher(on_change=self.waker.wake)
            if self.cfg.use_input:
                self.input = InputIdle()
            self._loop()
        except Exception:
            log.exception("erro inesperado")
            self.crashed = True
        finally:
            self.cam.close()
            if self.watcher:
                self.watcher.close()
            if self.ui:
                self.ui.quit()

    def _loop(self):
        CAMERA_FLAG.unlink(missing_ok=True)
        paused = "iniciando"
        next_analysis = 0.0
        camera_error_logged = -1e9
        while not self.stop.is_set():
            now = time.monotonic()
            self.housekeeping(now)
            reason = self.pause_reason(now)
            if reason != paused:
                if reason:
                    log.info("pausado: %s", reason)
                else:
                    log.info("vigiando (reconhecimento %s)",
                             "ligado" if self.recognition_active() else "desligado")
                paused = reason
            if reason:
                self.cam.close()
                self.set_blur(None)
                self.reset_presence(now)
                self.tracker.clear()
                self.sleep(self.pause_timeout(reason, now))
                continue
            if now < next_analysis:
                self.sleep(next_analysis - now)
                continue
            if not self.cam.is_open:
                try:
                    self.cam.open()
                except CameraError as e:
                    if now - camera_error_logged > 60:
                        log.warning("%s", e)
                        camera_error_logged = now
                    self.sleep(2)
                    continue
            frame = self.cam.fresh_frame()
            if frame is None:
                log.warning("a câmera parou de responder; reabrindo")
                self.cam.close()
                self.sleep(1)
                continue
            now = time.monotonic()
            if self.locked:
                self.process_locked(frame, now)
            else:
                self.process(frame, now)
            next_analysis = now + 1.0 / self.rate()

    def check_blind(self, frame, now):
        if not is_dark(frame):
            if self.blind:
                log.info("a imagem da câmera voltou")
                self.blind = False
            self.dark_since = None
            return False
        if self.dark_since is None:
            self.dark_since = now
        if now - self.dark_since >= 3 and not self.blind:
            self.blind = True
            log.warning("imagem da câmera escura: bloqueio e borrão pausados")
            if self.cfg.notifications:
                notify("Câmera sem imagem",
                       "Tampa da câmera fechada ou emissor IR desligado? "
                       "Bloqueio e borrão ficam pausados até a imagem voltar.")
        self.set_blur(None)
        self.alert = False
        self.last_present = self.last_looking = now
        return True

    def identify(self, frame, rows, tracks, now):
        """Confere quem é cada rosto: os novos na hora, o dono a cada 5 s."""
        cfg = self.cfg
        checks = 0
        for row, track in zip(rows, tracks):
            due = 5.0 if track.status == "dono" else 1.5
            if now - track.checked < due or checks >= 2:
                continue
            checks += 1
            track.checked = now
            emb = self.engine.embedding(frame, row)
            track.sim, track.label = self.gallery.match(emb)
            if self.learn_pending and now - self.unlocked_at < LEARN_WINDOW and len(rows) == 1:
                self.learn_pending = False
                if track.sim < cfg.owner_threshold:
                    # Acabou de desbloquear (Howdy ou senha) e só tem você na frente:
                    # guarda essa amostra para te reconhecer melhor da próxima vez.
                    self.gallery.add(emb, track.label or "sem_oculos", auto=True)
                    self.gallery.save()
                    self.file_mtimes = self.config_mtimes()
                    log.info("aprendi uma amostra nova do seu rosto (semelhança era %.2f)", track.sim)
                    track.status, track.low_hits = "dono", 0
                    continue
            if track.sim >= cfg.owner_threshold:
                track.status, track.low_hits = "dono", 0
            elif track.sim < cfg.stranger_threshold:
                # uma vez só pode ser ângulo ruim; duas seguidas (1,5 s) é outra pessoa
                track.low_hits += 1
                track.status = "estranho" if track.low_hits >= 2 else "?"
            else:
                track.low_hits = 0
            log.debug("rosto %s: semelhança %.2f (%s)", track.status, track.sim, track.label)

    def process(self, frame, now):
        cfg = self.cfg
        if self.check_blind(frame, now):
            return
        rows = self.engine.detect(frame)
        fw = frame.shape[1]
        rows = [r for r in rows if r[2] >= MIN_FACE_WIDTH * fw]
        tracks = self.tracker.update(rows, now)
        recog = self.recognition_active()
        if recog:
            self.identify(frame, rows, tracks, now)
        subject = next((i for i, t in enumerate(tracks) if t.status != "estranho"), None)
        strangers = sum(t.status == "estranho" for t in tracks)
        alone_stranger = subject is None and strangers > 0
        idle = self.input.seconds() if self.input else None
        typing = idle is not None and idle < cfg.input_seconds  # mexendo no teclado/mouse

        looking, why = False, "sem rosto"
        head_turned = False
        if subject is not None:
            self.last_present = now
            box = tuple(float(v) for v in rows[subject][:4])
            luma = region_luma(frame, box)
            self.face_box = box
            self.face_luma = luma if self.face_luma is None else 0.8 * self.face_luma + 0.2 * luma
            pose = self.engine.pose(frame, rows[subject])
            if pose is None:
                looking, why = True, ""  # achou o rosto mas não conseguiu medir: não pune
            else:
                skip_for_glasses = (recog and cfg.ignore_eyes_with_glasses
                                    and tracks[subject].label == "com_oculos")
                use_eyes = cfg.use_gaze and not pose.glare and not skip_for_glasses
                looking, why = self.attention.check(pose, use_eyes)
                if (looking and skip_for_glasses and cfg.use_gaze and not pose.glare
                        and not self.attention.check(pose, True)[0]):
                    # os olhos discordam da cabeça: confere já se você ainda está de óculos
                    tracks[subject].checked = -1e9
                if use_eyes and pose.blink >= BLINK_CLOSED:
                    if self.eyes_closed_since is None:
                        self.eyes_closed_since = now
                    if 0 < cfg.eyes_closed_after <= now - self.eyes_closed_since:
                        looking, why = False, "olhos fechados"
                else:
                    self.eyes_closed_since = None
                if log.isEnabledFor(logging.DEBUG):
                    feats = self.attention.features(pose)
                    log.debug("rostos=%d yaw=%+.0f pitch=%+.0f olhos=(%+.2f,%+.2f)%s -> %s",
                              len(rows), feats["yaw"], feats["pitch"], pose.gaze_h, pose.gaze_v,
                              "" if use_eyes else " [olhos ignorados]",
                              "olhando" if looking else f"não olhando ({why})")
        elif not alone_stranger and self.face_box is not None and self.face_luma:
            # No infravermelho a cabeça perto da câmera fica clara e o fundo escuro: se onde
            # estava o rosto continua claro, você só virou o rosto (de perfil ele some).
            head_turned = (now - self.last_present < TURNED_MAX and self.face_luma >= 25
                           and region_luma(frame, self.face_box) >= HEAD_STILL_RATIO * self.face_luma)
        if typing and not alone_stranger:
            # digitando/mexendo no mouse: você está aí, mesmo olhando o teclado ou fora da câmera
            if subject is None:
                self.last_present = now
            if not looking:
                log.debug("teclado/mouse em uso: conta como olhando")
            looking = True
        if subject is None and not typing:
            log.debug("sem você na frente (%d rosto(s) estranho(s))%s", strangers,
                      " — cabeça virada" if head_turned else "")
        self.looking_streak = self.looking_streak + 1 if looking else 0
        if looking:
            self.last_looking = now

        # outra pessoa olhando a tela junto com você
        others = 0
        if subject is not None:
            min_w = max(MIN_FACE_WIDTH * fw, OTHER_FACE_RATIO * rows[subject][2])
            for i, (row, track) in enumerate(zip(rows, tracks)):
                if i == subject or (recog and track.status == "dono"):
                    continue
                if row[2] >= min_w and abs(facing_ratio(row)) < FACING_MAX:
                    others += 1
        if not others:
            self.other_since = None
        elif self.other_since is None:
            self.other_since = now
        other_alert = self.other_since is not None and now - self.other_since >= 1.0

        # outra pessoa no seu lugar, sem você
        if not alone_stranger:
            self.stranger_since = None
        elif self.stranger_since is None:
            self.stranger_since = now

        self.alert = subject is None or not looking or other_alert
        if now < self.grace_until:
            self.set_blur(None)
            return

        absent_for = now - self.last_present
        limit = min(cfg.lock_after, cfg.stranger_lock_after) if alone_stranger else cfg.lock_after
        if cfg.lock and absent_for >= limit:
            self.set_blur(None)
            if now - self.lock_requested >= 5:
                self.lock_requested = now
                log.info("%s há %.0f s: bloqueando a tela",
                         "outra pessoa no seu lugar" if alone_stranger else "ninguém na frente",
                         absent_for)
                if not lock_screen():
                    log.warning("não consegui bloquear a tela")
                elif self.watcher:
                    self.watcher.resync()  # já pausa e solta a câmera para o Hello/Howdy
            return

        if other_alert and cfg.other_person == "avisar" and now - self.other_notified >= 60:
            self.other_notified = now
            notify("Tem outra pessoa olhando sua tela",
                   "Detectei mais um rosto virado para o notebook.")

        reason = None
        if cfg.blur and alone_stranger and now - self.stranger_since >= 1.0:
            reason = "estranho"  # clicar não tira esse
        elif cfg.blur and now >= self.snooze_until:
            if self.blur_reason == "olhar":
                # já está borrado: só tira depois de 2 análises seguidas olhando
                attention = self.looking_streak < 2
            else:
                attention = now - self.last_looking >= cfg.blur_after
            if attention:
                reason = "olhar"
            elif other_alert and cfg.other_person == "borrar":
                reason = "outra pessoa"
        if reason in ("olhar", "outra pessoa") and cfg.pause_blur_in_videos and self.video_playing(now):
            reason = None
        level = 0
        if reason == "estranho":
            level = self.darkening(absent_for, limit)
        elif reason == "olhar":
            # só virou o rosto: leve; saiu: forte, escurecendo até bloquear
            present = subject is not None or head_turned
            level = cfg.level_look if present else self.darkening(absent_for, cfg.lock_after)
        elif reason == "outra pessoa":
            level = cfg.level_other
        self.alert = self.alert or reason is not None
        self.set_blur(reason, level)

    def process_locked(self, frame, now):
        """Tela bloqueada + desbloqueio_ao_chegar: quando você volta, chama o Howdy."""
        if is_dark(frame):
            self.face_since = None
            return
        rows = [r for r in self.engine.detect(frame) if r[2] >= MIN_FACE_WIDTH * frame.shape[1]]
        tracks = self.tracker.update(rows, now)
        recog = self.recognition_active()
        if recog:
            self.identify(frame, rows, tracks, now)
        you = any(abs(facing_ratio(row)) < FACING_MAX and (track.status == "dono" or not recog)
                  for row, track in zip(rows, tracks))
        if not you:
            if not rows:
                if self.absent_since is None:
                    self.absent_since = now
                if now - self.absent_since >= 5:
                    self.wake_attempts = 0
            self.face_since = None
            return
        self.absent_since = None
        if self.face_since is None:
            self.face_since = now
        if (now - self.face_since >= 0.6 and self.wake_attempts < 2
                and now >= self.wake_cooldown):
            self.wake_attempts += 1
            self.wake_cooldown = now + 15
            self.face_since = None
            log.info("você chegou com a tela bloqueada: chamando o Howdy")
            self.cam.close()
            self.yield_until = now + 6
            self.sleep(0.3)
            # Ctrl abre a caixa de senha do GNOME sem digitar nada nela.
            sh("xdotool", "key", "Control_L")


# --------------------------------------------------------------------------- tela borrada

class Overlay:
    """Janela por cima de tudo com uma "foto" borrada da tela (X11), sem texto.

    A thread da câmera fala com o Tk por um pipe: o Tk só acorda quando chega um
    pedido (ou um sinal, via set_wakeup_fd), em vez de ficar conferindo toda hora."""

    def __init__(self, root, on_click):
        import tkinter as tk

        self.root = root
        self.on_click = on_click
        self.win = self.canvas = self.photo = self.shot = None
        self.bases, self.level, self.image_item = {}, 0, None
        self.toasts = queue.SimpleQueue()
        # Linux: pipe + createfilehandler (o Tk só acorda quando chega pedido).
        # Windows não tem createfilehandler: uma fila conferida 5x por segundo.
        self.use_pipe = not IS_WINDOWS and hasattr(root.tk, "createfilehandler")
        if self.use_pipe:
            self.read_fd, self.write_fd = os.pipe()
            for fd in (self.read_fd, self.write_fd):
                os.set_blocking(fd, False)
            root.tk.createfilehandler(self.read_fd, tk.READABLE, self._on_pipe)
        else:
            self.write_fd = None
            self.inbox = queue.SimpleQueue()
            root.after(200, self._poll)

    def request(self, level):
        """Chamado pela thread da câmera: 0 tira o borrão, 1..5 = nível (BLUR_LEVELS)."""
        self._send(str(int(level)).encode())

    def quit(self):
        """A thread da câmera terminou: encerra o Tk também."""
        self._send(b"q")

    def _send(self, byte):
        if not self.use_pipe:
            self.inbox.put(byte)
            return
        try:
            os.write(self.write_fd, byte)
        except BlockingIOError:
            pass

    def toast(self, title, body=""):
        """Aviso pequeno no canto da tela (usado no Windows)."""
        self.toasts.put((title, body))
        self._send(b"t")

    def _poll(self):
        data = b""
        while True:
            try:
                data += self.inbox.get_nowait()
            except queue.Empty:
                break
        if data:
            try:
                self._handle_bytes(data)
            except Exception:
                log.exception("erro ao mostrar/tirar o borrão")
        self.root.after(200, self._poll)

    def _on_pipe(self, fd, mask):
        # o Tk não protege este callback: um erro aqui derrubaria o serviço inteiro
        try:
            self._handle_pipe(fd)
        except Exception:
            log.exception("erro ao mostrar/tirar o borrão")

    def _handle_pipe(self, fd):
        try:
            data = os.read(fd, 4096)
        except BlockingIOError:
            return
        self._handle_bytes(data)

    def _handle_bytes(self, data):
        # bytes de sinal também caem aqui; o Python trata o sinal logo depois deste callback
        if b"t" in data:
            while True:
                try:
                    self.show_toast(*self.toasts.get_nowait())
                except queue.Empty:
                    break
        if b"q" in data:
            self.root.quit()
            return
        level = next((b - 48 for b in reversed(data) if 48 <= b <= 53), None)  # "0".."5"
        if level is None:
            return
        if level == 0:
            self._hide()
        elif self.win is None:
            self._show(level)
        else:
            self._set_level(level)

    def _render(self, level):
        """A foto da tela borrada no nível pedido (a foto é tirada uma vez só, ao aparecer)."""
        import tkinter as tk

        factor, brightness = BLUR_LEVELS[level]
        base = self.bases.get(factor)
        if base is None:
            h, w = self.shot.shape[:2]
            small = cv2.resize(self.shot, (max(1, w // factor), max(1, h // factor)),
                               interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 1.2)
            base = self.bases[factor] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(cv2.convertScaleAbs(base, alpha=brightness), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return tk.PhotoImage(data=f"P6 {w} {h} 255 ".encode() + rgb.tobytes(), format="PPM")

    def _show(self, level):
        import tkinter as tk

        with screen_grabber() as sct:
            whole = sct.monitors[0]
            self.shot = np.ascontiguousarray(np.asarray(sct.grab(whole))[:, :, :3])
        self.bases = {}
        h, w = self.shot.shape[:2]
        self.photo, self.level = self._render(level), level

        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.geometry(f"{w}x{h}+{whole['left']}+{whole['top']}")
        canvas = tk.Canvas(win, width=w, height=h, highlightthickness=0, bd=0, bg="black")
        canvas.pack()
        self.image_item = canvas.create_image(0, 0, anchor="nw", image=self.photo)
        canvas.bind("<Button-1>", self._clicked)
        win.attributes("-topmost", True)
        self.win, self.canvas = win, canvas
        self._fade(0.0)
        win.lift()

    def _set_level(self, level):
        if level == self.level:
            return
        self.photo, self.level = self._render(level), level
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def _fade(self, alpha):
        if self.win is None:
            return
        try:
            self.win.attributes("-alpha", alpha)
        except Exception:
            return
        if alpha < 1.0:
            self.root.after(16, self._fade, min(1.0, alpha + 0.2))

    def _hide(self):
        if self.win is not None:
            self.win.destroy()
        self.win = self.canvas = self.photo = self.shot = None
        self.bases, self.level = {}, 0

    def _clicked(self, _event):
        if self.on_click():
            self._hide()

    def show_toast(self, title, body="", seconds=5):
        import tkinter as tk

        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        frame = tk.Frame(win, bg="#202124", padx=16, pady=12)
        frame.pack()
        tk.Label(frame, text=title, fg="white", bg="#202124",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        if body:
            tk.Label(frame, text=body, fg="#d0d0d0", bg="#202124", font=("Segoe UI", 9),
                     wraplength=340, justify="left").pack(anchor="w")
        win.update_idletasks()
        x0, y0, w, h = laptop_monitor()
        win.geometry(f"+{x0 + w - win.winfo_width() - 24}+{y0 + h - win.winfo_height() - 72}")
        win.after(seconds * 1000, win.destroy)


# --------------------------------------------------------------------------- comandos

def cmd_run(args):
    import tkinter as tk

    cfg = Config()
    ensure_models("detector", "landmarker")
    device, source = windows_pick_camera(cfg) if IS_WINDOWS else resolve_device(cfg)
    log.info("câmera: %s (%s)", device, source)
    guard = Guard(cfg, device)
    if not guard.attention.calibrated:
        log.info("sem calibração: rode 'presence-guard calibrate' para ficar mais preciso")
    if not guard.gallery.enrolled:
        log.info("rosto não cadastrado: rode 'presence-guard cadastrar' para ele reconhecer você")
    x_init_threads()
    root = tk.Tk()
    root.withdraw()
    overlay = Overlay(root, on_click=guard.snooze)
    guard.ui = overlay
    if IS_WINDOWS:
        globals()["TOAST"] = overlay.toast
    worker = threading.Thread(target=guard.run, name="camera", daemon=True)

    def shutdown(*_):
        guard.shutdown()
        root.quit()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    if overlay.use_pipe:
        # o Tk dorme sem timers; um sinal escreve no pipe dele e o acorda para tratar
        signal.set_wakeup_fd(overlay.write_fd, warn_on_full_buffer=False)
    worker.start()
    try:
        root.mainloop()
    finally:
        guard.shutdown()
        worker.join(timeout=5)
    guard.engine.close()
    return 1 if guard.crashed else 0


def cmd_run_windows(args):
    """Só uma cópia por vez; o PID fica guardado para o `off` conseguir parar."""
    if pid_file_alive(PID_FILE):
        log.info("o vigia já está rodando")
        return 0
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    try:
        return cmd_run(args)
    finally:
        PID_FILE.unlink(missing_ok=True)


def windows_pick_camera(cfg):
    """Infravermelho do Windows Hello se abrir de verdade; senão a câmera normal."""
    device, source = resolve_device_windows(cfg)
    if device != WINRT_IR:
        return device, source
    probe = WindowsIRCamera()
    try:
        probe.open()
    except CameraError as e:
        log.warning("%s", e)
        cfg_rgb = Config()
        cfg_rgb.device = "rgb"
        device, other = resolve_device_windows(cfg_rgb)
        return device, other + " (a infravermelha não abriu)"
    finally:
        probe.close()
    return device, f"{source}, {'controle exclusivo' if probe.exclusive else 'compartilhada'}, emissor {probe.torch}"


CAL_TARGETS = (
    ("centro", 0.5, 0.5),
    ("canto superior esquerdo", 0.03, 0.04),
    ("canto superior direito", 0.97, 0.04),
    ("canto inferior direito", 0.97, 0.96),
    ("canto inferior esquerdo", 0.03, 0.96),
)
CAL_SETTLE = 1.3   # segundos para o olho chegar no ponto
CAL_COLLECT = 2.0  # segundos medindo


def summarize(samples):
    return {k: float(np.median([getattr(s, k) for s in samples]))
            for k in ("yaw", "pitch", "gaze_h", "gaze_v")}


def build_calibration(points):
    """points[0] é o centro; o resto, os cantos da tela."""
    center = points[0]
    rel = [{"yaw": wrap(p["yaw"] - center["yaw"]), "pitch": wrap(p["pitch"] - center["pitch"]),
            "gaze_h": p["gaze_h"], "gaze_v": p["gaze_v"]} for p in points]
    ranges = {}
    for key, min_half in (("yaw", 5.0), ("pitch", 5.0), ("gaze_h", 0.1), ("gaze_v", 0.1)):
        lo, hi = min(r[key] for r in rel), max(r[key] for r in rel)
        # quem calibra sem mexer os olhos teria uma faixa estreita demais
        mid = 0.0 if key in ("yaw", "pitch") else (lo + hi) / 2
        ranges[key] = [round(min(lo, mid - min_half), 3), round(max(hi, mid + min_half), 3)]
    return {"center": {"yaw": round(center["yaw"], 3), "pitch": round(center["pitch"], 3)},
            "ranges": ranges,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}


def largest_pose(engine, frame):
    rows = [r for r in engine.detect(frame) if r[2] >= MIN_FACE_WIDTH * frame.shape[1]]
    return engine.pose(frame, rows[0]) if rows else None


def cmd_calibrate(args):
    import tkinter as tk

    cfg = Config()
    with camera_hold():
        device, source = resolve_device(cfg)
        print(f"Câmera: {device} ({source})")
        cam = make_camera(device)
        cam.open()
        engine = FaceEngine()
        x, y, w, h = laptop_monitor()

        root = tk.Tk()
        root.title("Presence Guard - calibração")
        root.configure(bg="black", cursor="none")
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.update_idletasks()
        root.attributes("-fullscreen", True)
        canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        root.bind("<Escape>", lambda _e: root.destroy())
        root.focus_force()

        st = {"i": -1, "phase": "intro", "t0": time.monotonic(), "samples": [],
              "points": [], "retries": 0, "result": None}

        def show_text(title, sub=""):
            canvas.delete("all")
            canvas.create_text(w / 2, h / 2 - 30, text=title, fill="white", width=w * 0.8,
                               font=("Ubuntu", 28, "bold"), justify="center")
            if sub:
                canvas.create_text(w / 2, h / 2 + 40, text=sub, fill="#bbbbbb", width=w * 0.8,
                                   font=("Ubuntu", 15), justify="center")

        def show_dot(color):
            _, px, py = CAL_TARGETS[st["i"]]
            cx, cy, r = px * w, py * h, 16
            canvas.delete("all")
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="")
            canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="white", outline="")

        def next_target():
            st["i"] += 1
            st["retries"] = 0
            st["t0"] = time.monotonic()
            if st["i"] < len(CAL_TARGETS):
                st["phase"] = "settle"
                show_dot("#ff9500")
                return
            st["result"] = build_calibration(st["points"])
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CALIBRATION_FILE.write_text(json.dumps(st["result"], indent=2))
            st["phase"] = "finish"
            show_text("Pronto! Calibração salva.", "O serviço já passa a usar a nova calibração.")

        def step():
            now = time.monotonic()
            phase, elapsed = st["phase"], now - st["t0"]
            if phase == "intro":
                if elapsed >= 4:
                    next_target()
                else:
                    show_text("Calibração do olhar",
                              "Siga o ponto com os olhos, do jeito que você olha a tela "
                              "normalmente (pode mexer a cabeça).\n"
                              f"Começa em {math.ceil(4 - elapsed)} s   •   Esc cancela")
            elif phase == "settle" and elapsed >= CAL_SETTLE:
                st.update(phase="collect", t0=now, samples=[])
                show_dot("#ff3b30")
            elif phase == "collect":
                frame = cam.fresh_frame()
                pose = largest_pose(engine, frame) if frame is not None else None
                if pose is not None:
                    st["samples"].append(pose)
                if elapsed >= CAL_COLLECT:
                    if len(st["samples"]) >= 4:
                        st["points"].append(summarize(st["samples"]))
                        next_target()
                    elif st["retries"] < 2:
                        st.update(phase="retry", t0=now, retries=st["retries"] + 1)
                        show_text("Não encontrei seu rosto",
                                  "Fique de frente para o notebook, a uns 40-70 cm.")
                    else:
                        st.update(phase="failed", t0=now)
                        show_text("Não consegui ver seu rosto",
                                  "Rode 'presence-guard preview' para conferir a câmera.")
            elif phase == "retry" and elapsed >= 2:
                st.update(phase="settle", t0=now)
                show_dot("#ff9500")
            elif phase in ("finish", "failed") and elapsed >= 2.5:
                root.destroy()
                return
            root.after(30, step)

        root.after(100, step)
        root.mainloop()
        cam.close()
        engine.close()

    if st["result"]:
        r = st["result"]["ranges"]
        print("Calibração salva em", CALIBRATION_FILE)
        print(f"  cabeça: yaw {r['yaw'][0]:+.0f}..{r['yaw'][1]:+.0f}°, "
              f"pitch {r['pitch'][0]:+.0f}..{r['pitch'][1]:+.0f}°")
        return 0
    print("Calibração cancelada.")
    return 1


ENROLL_SECONDS = 8
ENROLL_MAX_PER_LABEL = 25


def collect_faces(cam, engine, seconds):
    """Amostras variadas do maior rosto na frente da câmera durante `seconds`."""
    kept, extra_faces = [], 0
    end = time.monotonic() + seconds
    while time.monotonic() < end and len(kept) < ENROLL_MAX_PER_LABEL:
        frame = cam.fresh_frame()
        if frame is None:
            break
        rows = [r for r in engine.detect(frame) if r[2] >= MIN_FACE_WIDTH * frame.shape[1]]
        if rows:
            extra_faces = max(extra_faces, len(rows) - 1)
            emb = engine.embedding(frame, rows[0])
            # pula amostras quase iguais às que já tem: queremos variedade
            if not kept or max(float(k @ emb) for k in kept) < 0.93:
                kept.append(emb)
        left = max(0, math.ceil(end - time.monotonic()))
        print(f"\r  amostras: {len(kept):2d}   {'rosto ok' if rows else 'procurando rosto...'}"
              f"   ({left} s)   ", end="", flush=True)
        time.sleep(0.15)
    print()
    if extra_faces:
        print("  atenção: tinha mais de um rosto na imagem; usei o maior.")
    return kept


def cmd_enroll(args):
    cfg = Config()
    gallery = Gallery() if args.adicionar else Gallery(empty=True)
    print("Cadastro do seu rosto (fica só neste computador, em", FACES_FILE, ")")
    print("Durante a captura, olhe para a tela e mexa a cabeça devagar: um pouco para os")
    print("lados, para cima e para baixo, mais perto e mais longe.\n")
    with camera_hold():
        device, source = resolve_device(cfg)
        print(f"Câmera: {device} ({source})")
        engine = FaceEngine(landmarker=False, recognizer=True)
        cam = make_camera(device)
        cam.open()
        try:
            for label, name in LABELS.items():
                answer = input(f"\n>> Fique {name.upper()} e aperte Enter "
                               f"(ou digite p para pular): ")
                if answer.strip().lower().startswith("p"):
                    continue
                samples = collect_faces(cam, engine, ENROLL_SECONDS)
                if len(samples) < 5:
                    print("  poucas amostras; mais uma vez (chegue um pouco mais perto e "
                          "mexa mais a cabeça).")
                    samples += collect_faces(cam, engine, ENROLL_SECONDS)
                for emb in samples:
                    gallery.add(emb, label)
                print(f"  {len(samples)} amostras {name}.")
            if not gallery.enrolled:
                print("\nNada foi cadastrado.")
                return 1
            gallery.save()
            input("\n>> Teste: olhe para a câmera (com ou sem óculos) e aperte Enter ")
            best = 0.0
            for _ in range(6):
                frame = cam.fresh_frame()
                rows = engine.detect(frame) if frame is not None else []
                if rows:
                    sim, label = gallery.match(engine.embedding(frame, rows[0]))
                    best = max(best, sim)
                time.sleep(0.15)
        finally:
            cam.close()
    counts = gallery.counts()
    print("\nCadastro salvo:", ", ".join(f"{n} {LABELS[k]}" for k, n in counts.items() if n))
    if best >= cfg.owner_threshold:
        print(f"Teste: reconheci você (semelhança {best:.2f}, precisa de {cfg.owner_threshold:.2f}).")
    else:
        print(f"Teste: semelhança {best:.2f}, abaixo de {cfg.owner_threshold:.2f}. "
              "Rode 'presence-guard cadastrar --adicionar' com mais luz/ângulos.")
    print("O serviço passa a usar o cadastro sozinho em alguns segundos.")
    return 0


def cmd_forget(args):
    FACES_FILE.unlink(missing_ok=True)
    print("Cadastro do rosto apagado. Qualquer rosto volta a contar como você.")
    return 0


def put_lines(img, lines):
    y = 24
    for text, color in lines:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 24


def cmd_preview(args):
    cfg = Config()
    white, green, red, yellow = (255, 255, 255), (80, 220, 80), (60, 60, 255), (0, 220, 255)
    with camera_hold():
        device, source = resolve_device(cfg)
        gallery = Gallery()
        recog = cfg.recognition and gallery.enrolled
        engine = FaceEngine(recognizer=recog)
        attention = Attention(cfg)
        cam = make_camera(device)
        cam.open()
        title = "Presence Guard - preview (q para sair)"
        try:
            while True:
                frame = cam.fresh_frame()
                if frame is None:
                    break
                fw = frame.shape[1]
                rows = [r for r in engine.detect(frame) if r[2] >= MIN_FACE_WIDTH * fw]
                scale = max(1.0, 640 / fw)
                vis = cv2.resize(frame, None, fx=scale, fy=scale)
                lines = [(f"camera {device} ({source})  brilho {frame.mean():.0f}"
                          + ("  ESCURA" if is_dark(frame) else ""), white),
                         (("calibrado" if attention.calibrated else "SEM calibracao") + "  |  "
                          + ("rosto cadastrado" if recog else "SEM cadastro de rosto"), white)]
                for i, row in enumerate(rows):
                    color, tag = white, f"rosto {i + 1}"
                    if recog:
                        sim, label = gallery.match(engine.embedding(frame, row))
                        if sim >= cfg.owner_threshold:
                            color, tag = green, f"VOCE {sim:.2f} ({label.replace('_', ' ')})"
                        elif sim < cfg.stranger_threshold:
                            color, tag = red, f"OUTRA PESSOA {sim:.2f}"
                        else:
                            color, tag = yellow, f"? {sim:.2f}"
                    x, y, w, h = (int(v * scale) for v in row[:4])
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis, tag, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1, cv2.LINE_AA)
                if rows:
                    pose = engine.pose(frame, rows[0])
                    if pose is None:
                        lines.append(("rosto achado, mas sem medir a cabeca", yellow))
                    else:
                        use_eyes = cfg.use_gaze and not pose.glare
                        ok, why = attention.check(pose, use_eyes)
                        feats = attention.features(pose)
                        lines += [(f"cabeca: yaw {feats['yaw']:+.0f}  pitch {feats['pitch']:+.0f}", white),
                                  (f"olhos: h {pose.gaze_h:+.2f}  v {pose.gaze_v:+.2f}  "
                                   f"fechados {pose.blink:.2f}"
                                   + ("  REFLEXO (olhos ignorados)" if pose.glare else ""), white),
                                  ("OLHANDO" if ok else f"NAO OLHANDO ({why})", green if ok else red)]
                else:
                    lines.append(("SEM ROSTO", red))
                put_lines(vis, lines)
                cv2.imshow(title, vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            cam.close()
            engine.close()
            cv2.destroyAllWindows()
    return 0


def service_active():
    if IS_WINDOWS:
        return pid_file_alive(SUPERVISOR_PID) or pid_file_alive(PID_FILE)
    r = sh("systemctl", "--user", "is-active", SERVICE)
    return bool(r and r.stdout.strip() == "active")


def cli_notify(title, body=""):
    """Aviso para comandos (no Windows o atalho Ctrl+Alt+P não tem terminal para mostrar)."""
    print(title + (f": {body}" if body else ""))
    if not IS_WINDOWS:
        notify(title, body)
        return
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        Overlay(root, on_click=lambda: True).show_toast(title, body, seconds=3)
        root.after(3000, root.destroy)
        root.mainloop()
    except Exception:
        pass


SUPERVISED_ARGS = ["run"]


def start_supervisor():
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(pythonw if pythonw.exists() else exe), str(Path(__file__).resolve()),
                      "supervise"], creationflags=flags, close_fds=True, cwd=str(DATA_DIR))


def stop_instances():
    for path in (SUPERVISOR_PID, PID_FILE):  # o supervisor primeiro, senão ele reinicia o vigia
        try:
            pid = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
        if process_alive(pid):
            terminate(pid)
        path.unlink(missing_ok=True)
    CAMERA_FLAG.unlink(missing_ok=True)


def cmd_supervise(args):
    """Windows: mantém o vigia rodando e reinicia se ele cair, até você desligar."""
    if DISABLED_FILE.exists() or pid_file_alive(SUPERVISOR_PID):
        return 0
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    SUPERVISOR_PID.write_text(str(os.getpid()))
    crashes = []
    try:
        while not DISABLED_FILE.exists():
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *SUPERVISED_ARGS],
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            code = child.wait()
            if code == 0 or DISABLED_FILE.exists():
                break
            now = time.monotonic()
            crashes = [t for t in crashes if now - t < 600] + [now]
            if len(crashes) > 5:
                log.error("o vigia caiu 5 vezes em 10 minutos; parei de reiniciar (veja o log)")
                break
            log.warning("o vigia parou com erro (código %s); reiniciando em 5 s", code)
            time.sleep(5)
    finally:
        SUPERVISOR_PID.unlink(missing_ok=True)
    return 0


def cmd_cameras(args):
    if not IS_WINDOWS:
        print("No Linux: v4l2-ctl --list-devices  (o padrão é a mesma câmera do Howdy)")
        return 0
    print("Câmeras infravermelhas que o Windows mostra para os apps (API do Windows):")
    try:
        names = run_in_mta(windows_ir_sources)
    except Exception as e:
        names = []
        print("  erro na API de câmera do Windows:", e)
    for name in names:
        print("  -", name)
    if names:
        probe = WindowsIRCamera()
        try:
            probe.open()
            frame = probe.fresh_frame()
            light = f"brilho {frame.mean():.0f}" if frame is not None else "sem imagem"
            print(f"  abriu: {'controle exclusivo' if probe.exclusive else 'compartilhada'}, "
                  f"emissor IR {probe.torch}, {light}")
        except CameraError as e:
            print("  NÃO abriu:", e)
        finally:
            probe.close()
    else:
        print("  nenhuma. Se a 'Segurança de Entrada Aprimorada' do Windows Hello estiver ligada,")
        print("  o Windows não empresta a câmera infravermelha para nenhum outro app.")
    print("Câmeras pelo OpenCV (a luz da câmera pode piscar):")
    for index, gray in windows_cameras():
        print(f"  câmera {index}: {'infravermelho' if gray else 'normal (colorida)'}")
    device, source = windows_pick_camera(Config())
    print(f"Vou usar: {device} ({source}). Para forçar: dispositivo = ir | rgb | <número> no config.ini")
    return 0


def cmd_on(args):
    if IS_WINDOWS:
        DISABLED_FILE.unlink(missing_ok=True)
        if not service_active():
            start_supervisor()
        cli_notify("Presence Guard ligado", "Bloqueio e borrão automáticos ativos.")
        return 0
    r = sh("systemctl", "--user", "enable", "--now", SERVICE, timeout=15)
    if not r or r.returncode != 0:
        print("não consegui ligar:", r.stderr.strip() if r else "systemctl indisponível")
        return 1
    notify("Presence Guard ligado", "Bloqueio e borrão automáticos ativos.")
    print("ligado (e liga sozinho quando você entrar no Ubuntu)")
    return 0


def cmd_off(args):
    if IS_WINDOWS:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        DISABLED_FILE.touch()
        stop_instances()
        cli_notify("Presence Guard desligado", "Câmera livre e sem gasto de bateria. "
                                               "Ctrl+Alt+P ou presence-guard on religa.")
        return 0
    r = sh("systemctl", "--user", "disable", "--now", SERVICE, timeout=15)
    if not r or r.returncode != 0:
        print("não consegui desligar:", r.stderr.strip() if r else "systemctl indisponível")
        return 1
    notify("Presence Guard desligado", "Câmera livre e sem gasto de bateria. "
                                       "Para religar: presence-guard on")
    print("desligado (continua desligado até você rodar: presence-guard on)")
    return 0


def cmd_toggle(args):
    return cmd_off(args) if service_active() else cmd_on(args)


def cmd_log(args):
    if IS_WINDOWS:
        try:
            lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            print("ainda não tem log em", LOG_FILE)
            return 0
        print("\n".join(lines[-80:]))
        return 0
    os.execvp("journalctl", ["journalctl", "--user", "-u", SERVICE, "-n", "80",
                             "--no-pager", "-o", "short"])


def cmd_debug(args):
    was_active = service_active()
    if was_active:
        if IS_WINDOWS:
            stop_instances()
        else:
            sh("systemctl", "--user", "stop", SERVICE, timeout=15)
    logging.getLogger().setLevel(logging.DEBUG)
    print("Rodando com log detalhado. Ctrl+C para sair.", flush=True)
    try:
        return cmd_run(args)
    finally:
        if was_active:
            if IS_WINDOWS:
                if not DISABLED_FILE.exists():
                    start_supervisor()
            else:
                sh("systemctl", "--user", "start", SERVICE, timeout=15)


def cmd_status(args):
    cfg = Config()
    if IS_WINDOWS:
        print("ligado:        ", "sim" if service_active() else "não",
              "(desligado até rodar: presence-guard on)" if DISABLED_FILE.exists() else "")
        try:
            cams = json.loads(CAMERAS_FILE.read_text())
            names = ", ".join(f"{i} ({'IR' if g else 'normal'})" for i, g in cams)
        except (OSError, ValueError):
            names = "ainda não procurei (presence-guard cameras)"
        print("câmera:        ", cfg.device, "->", names)
    else:
        enabled = sh("systemctl", "--user", "is-enabled", SERVICE)
        print("ligado:        ", "sim" if service_active() else "não",
              "(liga sozinho ao entrar)" if enabled and enabled.stdout.strip() == "enabled" else "")
        print("câmera:        ", cfg.device if cfg.device.lower() != "auto"
              else f"auto -> {howdy_device() or 'detectar ao iniciar'}")
    print("calibração:    ", json.loads(CALIBRATION_FILE.read_text()).get("created", "sim")
          if CALIBRATION_FILE.exists() else "não (rode: presence-guard calibrate)")
    gallery = Gallery()
    if gallery.enrolled:
        counts = gallery.counts()
        auto = int(gallery.auto.sum())
        print("rosto:         ", ", ".join(f"{n} amostras {LABELS[k]}" for k, n in counts.items() if n)
              + (f" + {auto} aprendidas" if auto else "")
              + ("" if cfg.recognition else " (reconhecimento desligado no config)"))
    else:
        print("rosto:          não cadastrado (rode: presence-guard cadastrar)")
    missing = [n for n in MODELS if not model_path(n).exists()]
    print("modelos:       ", "ok" if not missing else "faltando " + ", ".join(missing))
    if IS_WINDOWS:
        print("log:           ", LOG_FILE)
        return 0
    pam = [p for p in glob.glob("/etc/pam.d/*")
           if "presence-guard-yield" in Path(p).read_text(errors="ignore")]
    print("integração PAM:", ", ".join(pam) if pam else "não instalada")
    return 0


def cmd_download_models(args):
    ensure_models(*MODELS)
    print("modelos ok em", DATA_DIR)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=APP, description="Bloqueia o computador quando você sai e borra a tela "
                              "quando você não está olhando.")
    parser.add_argument("--debug", action="store_true", help="mostra cada análise no log")
    sub = parser.add_subparsers(dest="command")
    commands = (("run", cmd_run_windows if IS_WINDOWS else cmd_run), ("on", cmd_on),
                ("off", cmd_off), ("toggle", cmd_toggle), ("supervise", cmd_supervise),
                ("cameras", cmd_cameras),
                ("calibrate", cmd_calibrate), ("preview", cmd_preview), ("status", cmd_status),
                ("log", cmd_log), ("debug", cmd_debug), ("esquecer", cmd_forget),
                ("download-models", cmd_download_models))
    for name, func in commands:
        sub.add_parser(name).set_defaults(func=func)
    enroll = sub.add_parser("cadastrar", aliases=["enroll"],
                            help="grava o seu rosto com e sem óculos")
    enroll.add_argument("--adicionar", action="store_true",
                        help="soma ao cadastro atual em vez de começar do zero")
    enroll.set_defaults(func=cmd_enroll)
    args = parser.parse_args(argv)
    level = logging.DEBUG if args.debug else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(message)s"
    if IS_WINDOWS and args.command in (None, "run", "supervise"):
        from logging.handlers import RotatingFileHandler

        DATA_DIR.mkdir(parents=True, exist_ok=True)  # pythonw não tem terminal: vai para o arquivo
        handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=1, encoding="utf-8")
        logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S", handlers=[handler])
    else:
        logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    func = getattr(args, "func", cmd_run_windows if IS_WINDOWS else cmd_run)
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
