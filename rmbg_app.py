"""
AI Background Remover - Cross-platform Application (CustomTkinter UI)
Приложение для удаления фона с изображений с помощью AI-модели BiRefNet.

Работает на macOS и Windows.

Author: Stepan Andrushkevich
Version: 1.2
License: MIT (application code), Apache 2.0 (BiRefNet model)
"""

import io
import os
import sys
import logging
import platform
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageTk
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

import customtkinter as ctk
from tkinter import filedialog, Canvas

# ---------------------------------------------------------------------------
# Логирование (кроссплатформенное)
# ---------------------------------------------------------------------------

_system = platform.system()
if _system == "Darwin":
    LOG_DIR = os.path.expanduser("~/Library/Logs")
elif _system == "Windows":
    LOG_DIR = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")), "AI-Background-Remover"
    )
else:
    LOG_DIR = os.path.expanduser("~/.local/share/AI-Background-Remover")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "AI-Background-Remover.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("AIBGApp")

# ---------------------------------------------------------------------------
# Константы приложения
# ---------------------------------------------------------------------------

APP_NAME = "AI Background remover"
APP_VERSION = "1.2"
WINDOW_WIDTH = 960                  # Ширина окна (px)
WINDOW_HEIGHT = 720                 # Высота окна (px)
MODEL_ID = "ZhengPeng7/BiRefNet"    # HuggingFace ID модели для удаления фона
INPUT_RESOLUTION = (1024, 1024)     # Разрешение входного тензора для модели
DEFAULT_THRESHOLD = 128             # Порог по умолчанию (0-255)
DEFAULT_FEATHER = 0                 # Размытие краёв (0 = выкл)
DEFAULT_ERODE = 0                   # Erode/Dilate (0 = выкл, <0 = erode, >0 = dilate)
SUPPORTED_FORMATS = (".png", ".jpeg", ".jpg", ".webp", ".bmp", ".tiff")
CHECKERBOARD_LIGHT = "#FFFFFF"      # Светлый цвет шахматки (прозрачность)
CHECKERBOARD_DARK = "#C8C8C8"       # Тёмный цвет шахматки
CHECKERBOARD_SIZE = 10              # Размер клетки шахматки (px)
PREVIEW_MAX = 400                   # Макс. размер превью (px)
DEBOUNCE_MS = 100                   # Задержка обновления при движении слайдера (мс)


# ---------------------------------------------------------------------------
# Дизайн-токены (из Figma)
# ---------------------------------------------------------------------------

class Colors:
    """Все цвета UI из Figma-макета."""
    BG = "#eef1f8"                      # Фон приложения
    CARD = "#ffffff"                    # Фон карточек (белые блоки)
    TEXT_PRIMARY = "#121e2e"            # Основной текст (заголовки)
    TEXT_SECONDARY = "#5d6573"          # Вторичный текст (подписи, кнопки)
    TEXT_MUTED = "#b1b6c1"             # Приглушённый текст (футер)
    SURFACE_NEUTRAL = "#eef1f8"        # Фон нейтральных элементов (кнопка Open, тег Loading)
    SURFACE_NEUTRAL_HOVER = "#cfd3dc"  # Ховер нейтральных элементов
    PRIMARY = "#8800ff"                # Основной фиолетовый (слайдер, кнопка Save, тег Done)
    PRIMARY_HOVER = "#7124ce"          # Ховер фиолетового
    LOADING_BG = "#e7d1ff"             # Фон тега Loading (светло-фиолетовый)
    LOADING_BORDER = "#8800ff"         # Обводка тега Loading (фиолетовый)
    LOADING_TEXT = "#8800ff"           # Текст тега Loading (фиолетовый)
    SUCCESS_LIGHT_BG = "#b3f3d1"       # Фон тега Ready
    SUCCESS_LIGHT_TEXT = "#26ab5d"      # Текст тега Ready
    SUCCESS_SOLID_BG = "#00d560"       # Фон тега Image saved
    SUCCESS_SOLID_TEXT = "#eef1f8"     # Текст тега Image saved
    DANGER_BG = "#ffd1d9"             # Фон тега Error
    DANGER_TEXT = "#ff003c"            # Текст тега Error
    BORDER = "#b1b6c1"                # Цвет пунктирной рамки панелей
    PRIMARY_TEXT = "#eef1f8"           # Текст на фиолетовом фоне


CARD_RADIUS = 40        # Скругление карточек (px)
BUTTON_RADIUS = 8       # Скругление кнопок (px)
TAG_RADIUS = 999        # Скругление тегов (pill / капсула)
CARD_PADDING = 24       # Внутренний отступ карточек (px)
CARD_GAP = 2            # Зазор между карточками (px)

# Стили тегов статуса: (фон, цвет текста, текст)
TAG_STYLES = {
    "loading": (Colors.SURFACE_NEUTRAL, Colors.TEXT_SECONDARY, "Loading"),
    "ready": (Colors.SUCCESS_LIGHT_BG, Colors.SUCCESS_LIGHT_TEXT, "Ready"),
    "error": (Colors.DANGER_BG, Colors.DANGER_TEXT, "Error"),
    "done": (Colors.PRIMARY, Colors.PRIMARY_TEXT, "Done"),
    "saved": (Colors.SUCCESS_SOLID_BG, Colors.SUCCESS_SOLID_TEXT, "Image saved"),
}


# ---------------------------------------------------------------------------
# Вспомогательные функции (бизнес-логика)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Определяет устройство для инференса: MPS (Mac), CUDA (Nvidia) или CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_snapshot(base_dir: str) -> Optional[str]:
    """Находит путь к snapshot внутри кеша HuggingFace модели."""
    snapshots_dir = os.path.join(base_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        hashes = [h for h in os.listdir(snapshots_dir) if not h.startswith(".")]
        if hashes:
            return os.path.join(snapshots_dir, hashes[0])
    return None


def get_model_path() -> str:
    """Определяет путь к модели: сначала рядом со скриптом, потом кеш HF, потом скачивание."""
    # 1. Рядом со скриптом (встроена в .app / PyInstaller билд)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_cache = os.path.join(script_dir, "models--ZhengPeng7--BiRefNet")
    if os.path.isdir(local_cache):
        return _resolve_snapshot(local_cache) or local_cache

    # 2. Кеш HuggingFace (уже скачана ранее)
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    hf_cache = os.path.join(hf_home, "hub", "models--ZhengPeng7--BiRefNet")
    if os.path.isdir(hf_cache):
        return _resolve_snapshot(hf_cache) or hf_cache

    # 3. Скачать из HuggingFace Hub
    return MODEL_ID


def build_checkerboard(width: int, height: int) -> Image.Image:
    """Строит шахматный паттерн для отображения прозрачности."""
    img = Image.new("RGB", (width, height), CHECKERBOARD_LIGHT)
    draw = ImageDraw.Draw(img)
    for y in range(0, height, CHECKERBOARD_SIZE):
        for x in range(0, width, CHECKERBOARD_SIZE):
            if (x // CHECKERBOARD_SIZE + y // CHECKERBOARD_SIZE) % 2 == 0:
                draw.rectangle(
                    [x, y, x + CHECKERBOARD_SIZE - 1, y + CHECKERBOARD_SIZE - 1],
                    fill=CHECKERBOARD_DARK,
                )
    return img


def fit_thumbnail(image: Image.Image, max_size: int = PREVIEW_MAX) -> Image.Image:
    """Уменьшает изображение до max_size, сохраняя пропорции."""
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image


def apply_mask(
    original: Image.Image,
    matte: np.ndarray,
    threshold: int = 128,
    soft_alpha: int = 0,
    erode_dilate: int = 0,
    feather: int = 0,
    refinement: int = 0,
    invert: bool = False,
) -> Image.Image:
    """
    Применяет маску к изображению с опциональными параметрами обработки.

    Args:
        original: Исходное изображение.
        matte: Маска прозрачности (float32, 0-255).
        threshold: Порог бинаризации (0-255).
        soft_alpha: 0-100 — степень мягкости маски (0=бинарная, 100=полностью soft).
        erode_dilate: -10..+10 — сужение (−) или расширение (+) маски в пикселях.
        feather: 0-20 — радиус размытия краёв маски (Gaussian blur).
        refinement: 0-30 — радиус guided filter для уточнения краёв.
        invert: True — инвертировать маску (вырезать объект, оставить фон).
    """
    # 1. Базовая маска: blend между бинарной и soft
    binary = np.where(matte > threshold, 255.0, 0.0)
    soft = np.clip(matte, 0, 255).astype(np.float64)
    blend = soft_alpha / 100.0
    alpha_f = binary * (1.0 - blend) + soft * blend
    alpha_arr = np.clip(alpha_f, 0, 255).astype(np.uint8)

    # 2. Erode / Dilate
    if erode_dilate != 0:
        binary_mask = alpha_arr > 127
        iterations = abs(erode_dilate)
        if erode_dilate < 0:
            binary_mask = binary_erosion(binary_mask, iterations=iterations)
        else:
            binary_mask = binary_dilation(binary_mask, iterations=iterations)
        # Сохраняем плавные значения внутри маски
        erode_mask = binary_mask.astype(np.float32)
        alpha_arr = (alpha_f * erode_mask).clip(0, 255).astype(np.uint8)

    # 3. Edge refinement (guided filter)
    if refinement > 0:
        try:
            from cv2.ximgproc import guidedFilter
            guide = np.array(original.convert("RGB"), dtype=np.float32) / 255.0
            src = alpha_arr.astype(np.float32) / 255.0
            refined = guidedFilter(guide, src, radius=refinement, eps=1e-3)
            alpha_arr = np.clip(refined * 255, 0, 255).astype(np.uint8)
        except ImportError:
            logger.info("Edge refinement unavailable (opencv-contrib not installed)")

    # 4. Feather (Gaussian blur краёв)
    if feather > 0:
        alpha_pil = Image.fromarray(alpha_arr)
        alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=feather))
        alpha_arr = np.array(alpha_pil)

    # 5. Инверсия маски
    if invert:
        alpha_arr = 255 - alpha_arr

    # 6. Собираем результат
    result = original.copy().convert("RGBA")
    alpha = Image.fromarray(alpha_arr)
    if alpha.size != result.size:
        alpha = alpha.resize(result.size, Image.LANCZOS)
    result.putalpha(alpha)
    return result


# ---------------------------------------------------------------------------
# Препроцессинг изображения для модели (ImageNet нормализация)
# ---------------------------------------------------------------------------

_preprocess = transforms.Compose(
    [
        transforms.Resize(INPUT_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ---------------------------------------------------------------------------
# Кастомные виджеты
# ---------------------------------------------------------------------------

class CustomSlider(Canvas):
    """
    Кастомный слайдер на Canvas с PNG-текстурами из папки textures/.
    Элементы:
    - Левый трек (фиолетовый): left cap + middle (растяг.) + right cap
    - Thumb (ползунок): PNG-капсула 48×96 → масштабируется до экранного размера
    - Правый трек (серый): left cap + middle (растяг.) + right cap
    """

    DISPLAY_THUMB_W = 12    # Ширина ползунка на экране (px, из Figma)
    DISPLAY_THUMB_H = 24    # Высота ползунка на экране (px, из Figma)
    TRACK_H = 4             # Высота трека на экране (px, из Figma)
    CAP_W = 4               # Ширина скруглённого конца трека (px)
    GAP = 3                 # Зазор между треком и ползунком (px)

    def __init__(self, master, from_=0, to=255, value=128, command=None, **kwargs):
        super().__init__(
            master, height=self.DISPLAY_THUMB_H, highlightthickness=0, bd=0,
            bg=Colors.CARD, cursor="hand2", **kwargs,
        )
        self._min = from_
        self._max = to
        self._value = value
        self._command = command
        self._dragging = False

        # --- Загрузка и предварительный ресайз PNG-текстур (один раз) ---
        tex_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")

        # Thumb: 48×96 → 12×24 (LANCZOS, максимальное качество)
        self._thumb_photo = ImageTk.PhotoImage(
            Image.open(os.path.join(tex_dir, "slider-button.png"))
            .resize((self.DISPLAY_THUMB_W, self.DISPLAY_THUMB_H), Image.LANCZOS)
        )

        # Caps треков: 8×16 → 4×4 (маленькие скруглённые концы)
        self._active_left = Image.open(os.path.join(tex_dir, "slider-active-left.png"))
        self._active_mid = Image.open(os.path.join(tex_dir, "slider-active-middle.png"))
        self._active_right = Image.open(os.path.join(tex_dir, "slider-active-right.png"))
        self._normal_left = Image.open(os.path.join(tex_dir, "slider-normal.png"))
        self._normal_mid = Image.open(os.path.join(tex_dir, "slider-normal-middle.png"))
        self._normal_right = Image.open(os.path.join(tex_dir, "slider-normal-right.png"))

        # Пре-ресайз caps (статичны, не меняются)
        self._act_left_photo = ImageTk.PhotoImage(
            self._active_left.resize((self.CAP_W, self.TRACK_H), Image.LANCZOS))
        self._act_right_photo = ImageTk.PhotoImage(
            self._active_right.resize((self.CAP_W, self.TRACK_H), Image.LANCZOS))
        self._norm_left_photo = ImageTk.PhotoImage(
            self._normal_left.resize((self.CAP_W, self.TRACK_H), Image.LANCZOS))
        self._norm_right_photo = ImageTk.PhotoImage(
            self._normal_right.resize((self.CAP_W, self.TRACK_H), Image.LANCZOS))

        # Кеш динамических PhotoImage (middle-растяжки, пересоздаются при перерисовке)
        self._dyn_photos = []

        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = max(self._min, min(self._max, val))
        self._draw()

    def _value_to_x(self, value):
        """Преобразует значение слайдера в X-координату центра thumb."""
        w = self.winfo_width()
        half = self.DISPLAY_THUMB_W / 2
        usable = w - self.DISPLAY_THUMB_W
        if self._max == self._min:
            return half
        ratio = (value - self._min) / (self._max - self._min)
        return half + ratio * usable

    def _x_to_value(self, x):
        """Преобразует X-координату мыши в значение слайдера."""
        w = self.winfo_width()
        half = self.DISPLAY_THUMB_W / 2
        usable = w - self.DISPLAY_THUMB_W
        if usable <= 0:
            return self._min
        ratio = max(0, min(1, (x - half) / usable))
        return int(self._min + ratio * (self._max - self._min))

    def _on_press(self, event):
        self._dragging = True
        self._update_from_mouse(event.x)

    def _on_drag(self, event):
        if self._dragging:
            self._update_from_mouse(event.x)

    def _on_release(self, event):
        self._dragging = False

    def _update_from_mouse(self, x):
        new_val = self._x_to_value(x)
        if new_val != self._value:
            self._value = new_val
            self._draw()
            if self._command:
                self._command(self._value)

    def _draw(self, event=None):
        """Перерисовывает слайдер из PNG-текстур."""
        self.delete("all")
        self._dyn_photos.clear()
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            return

        thumb_cx = self._value_to_x(self._value)
        cy = h / 2
        half_thumb = self.DISPLAY_THUMB_W / 2
        cap_w = self.CAP_W

        # --- Фиолетовый трек слева от ползунка ---
        active_end = int(thumb_cx - half_thumb - self.GAP)
        if active_end > cap_w * 2:
            # Левый cap (пре-ресайзнутый)
            self.create_image(0, cy, image=self._act_left_photo, anchor="w")
            # Средняя часть (растягивается под текущую ширину)
            mid_w = active_end - cap_w * 2
            if mid_w > 0:
                p = ImageTk.PhotoImage(
                    self._active_mid.resize((mid_w, self.TRACK_H), Image.LANCZOS))
                self._dyn_photos.append(p)
                self.create_image(cap_w, cy, image=p, anchor="w")
            # Правый cap
            self.create_image(active_end - cap_w, cy, image=self._act_right_photo, anchor="w")

        # --- Серый трек справа от ползунка ---
        inactive_start = int(thumb_cx + half_thumb + self.GAP)
        inactive_len = w - inactive_start
        if inactive_len > cap_w * 2:
            # Левый cap
            self.create_image(inactive_start, cy, image=self._norm_left_photo, anchor="w")
            # Средняя часть (растягивается)
            mid_w = inactive_len - cap_w * 2
            if mid_w > 0:
                p = ImageTk.PhotoImage(
                    self._normal_mid.resize((mid_w, self.TRACK_H), Image.LANCZOS))
                self._dyn_photos.append(p)
                self.create_image(inactive_start + cap_w, cy, image=p, anchor="w")
            # Правый cap
            self.create_image(w - cap_w, cy, image=self._norm_right_photo, anchor="w")

        # --- Ползунок (пре-ресайзнутый PNG) ---
        self.create_image(int(thumb_cx), int(cy), image=self._thumb_photo, anchor="center")


class ToggleSwitch(Canvas):
    """
    Кастомный toggle switch на Canvas с PNG-текстурами.
    Файлы: textures/switch-on.png, textures/switch-off.png
    Отображаемый размер: 40×24
    """

    WIDTH = 40
    HEIGHT = 24

    def __init__(self, master, command=None, **kwargs):
        super().__init__(
            master, width=self.WIDTH, height=self.HEIGHT,
            highlightthickness=0, bd=0, bg=Colors.CARD,
            cursor="hand2", **kwargs,
        )
        self._on = False
        self._command = command

        # Загружаем и ресайзим PNG-текстуры
        tex_dir = os.path.join(
            getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))),
            "textures",
        )
        self._img_on = ImageTk.PhotoImage(
            Image.open(os.path.join(tex_dir, "switch-on.png"))
            .resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)
        )
        self._img_off = ImageTk.PhotoImage(
            Image.open(os.path.join(tex_dir, "switch-off.png"))
            .resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)
        )

        self.bind("<Button-1>", self._toggle)
        self._draw()

    @property
    def value(self):
        return self._on

    @value.setter
    def value(self, val):
        self._on = bool(val)
        self._draw()

    def _toggle(self, event=None):
        self._on = not self._on
        self._draw()
        if self._command:
            self._command(self._on)

    def _draw(self, event=None):
        self.delete("all")
        img = self._img_on if self._on else self._img_off
        self.create_image(self.WIDTH // 2, self.HEIGHT // 2, image=img, anchor="center")


class DashedPanel(Canvas):
    """
    Панель для отображения изображения с пунктирной рамкой.
    Используется для оригинала (слева) и результата (справа).
    Изображение масштабируется, чтобы заполнить панель (object-contain).
    """

    def __init__(self, master, **kwargs):
        super().__init__(
            master, highlightthickness=0, bd=0, bg=Colors.CARD, **kwargs,
        )
        self._photo = None       # ImageTk.PhotoImage (ссылка, чтобы GC не удалил)
        self._pil_image = None   # Исходное PIL изображение
        self.bind("<Configure>", self._on_resize)

    def set_image(self, pil_image):
        """Устанавливает изображение для отображения."""
        self._pil_image = pil_image
        self._redraw()

    def clear(self):
        """Очищает панель."""
        self._pil_image = None
        self._photo = None
        self._redraw()

    def _on_resize(self, event=None):
        self._redraw()

    def _redraw(self):
        """Перерисовывает панель: изображение + пунктирная рамка."""
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or h <= 1:
            return

        # --- Изображение (масштабируется под размер панели) ---
        if self._pil_image is not None:
            img = self._pil_image.copy()
            iw, ih = img.size
            area_w, area_h = w - 4, h - 4  # 2px отступ от рамки
            scale = min(area_w / iw, area_h / ih)
            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.create_image(w / 2, h / 2, image=self._photo, anchor="center")

        # --- Пунктирная рамка со скруглёнными углами ---
        r = 8  # Радиус скругления рамки
        self._draw_dashed_rounded_rect(1, 1, w - 1, h - 1, r)

    def _draw_dashed_rounded_rect(self, x1, y1, x2, y2, r):
        """Рисует пунктирный прямоугольник со скруглёнными углами."""
        color = Colors.BORDER
        dash = (6, 4)  # Длина штриха и пробела
        lw = 1
        # Стороны
        self.create_line(x1 + r, y1, x2 - r, y1, fill=color, dash=dash, width=lw)  # верх
        self.create_line(x2, y1 + r, x2, y2 - r, fill=color, dash=dash, width=lw)  # право
        self.create_line(x2 - r, y2, x1 + r, y2, fill=color, dash=dash, width=lw)  # низ
        self.create_line(x1, y2 - r, x1, y1 + r, fill=color, dash=dash, width=lw)  # лево
        # Скруглённые углы
        self.create_arc(
            x1, y1, x1 + 2 * r, y1 + 2 * r,
            start=90, extent=90, style="arc", outline=color, dash=dash, width=lw,
        )  # верхний левый
        self.create_arc(
            x2 - 2 * r, y1, x2, y1 + 2 * r,
            start=0, extent=90, style="arc", outline=color, dash=dash, width=lw,
        )  # верхний правый
        self.create_arc(
            x2 - 2 * r, y2 - 2 * r, x2, y2,
            start=270, extent=90, style="arc", outline=color, dash=dash, width=lw,
        )  # нижний правый
        self.create_arc(
            x1, y2 - 2 * r, x1 + 2 * r, y2,
            start=180, extent=90, style="arc", outline=color, dash=dash, width=lw,
        )  # нижний левый


# ---------------------------------------------------------------------------
# Главное приложение
# ---------------------------------------------------------------------------

class AIBGApp:
    def __init__(self):
        # ── Состояние приложения ──────────────────────────────────
        self.model = None                                    # AI-модель BiRefNet
        self.device = get_device()                           # Устройство (mps/cuda/cpu)
        self.original_image: Optional[Image.Image] = None    # Загруженное изображение
        self.alpha_matte: Optional[np.ndarray] = None        # Маска прозрачности (результат модели)
        self._processing = False                             # Флаг: идёт обработка
        self._original_filepath = ""                         # Путь к исходному файлу
        self._threshold = DEFAULT_THRESHOLD                  # Текущий порог
        self._feather = DEFAULT_FEATHER                      # Размытие краёв
        self._erode_dilate = DEFAULT_ERODE                   # Erode/Dilate
        self._soft_alpha = 0                                 # Soft alpha blend (0=binary, 100=full soft)
        self._refinement = 0                                 # Edge refinement radius (0=off)
        self._only_mask = False                              # Показывать только маску
        self._invert = False                                 # Инвертировать маску
        self._debounce_id = None                             # ID таймера debounce слайдера
        self._save_btn_shown = False                         # Флаг: кнопка Save уже показана

        # ── Окно приложения ───────────────────────────────────────
        ctk.set_appearance_mode("light")
        self.root = ctk.CTk()
        self.root.title(APP_NAME)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.configure(fg_color=Colors.BG)

        # Горячие клавиши: Cmd+O / Ctrl+O — открыть, Cmd+S / Ctrl+S — сохранить
        mod = "Command" if _system == "Darwin" else "Control"
        self.root.bind(f"<{mod}-o>", lambda e: self._open_image())
        self.root.bind(f"<{mod}-s>", lambda e: self._save_result())

        # ── Шрифты ────────────────────────────────────────────────
        self.font_title = ctk.CTkFont(family="Inter", size=20)   # Заголовок (H6)
        self.font_body = ctk.CTkFont(family="Inter", size=16)    # Текст кнопок (H7)
        self.font_small = ctk.CTkFont(family="Inter", size=12)   # Теги, подписи (H8)

        # ── Сборка UI ─────────────────────────────────────────────
        self._build_ui()

        # ── Загрузка модели в фоне ────────────────────────────────
        self._model_loading = True
        self._animate_loading()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _animate_loading(self):
        """Обновляет тег загрузки — считает размер скачанных файлов модели."""
        if not self._model_loading:
            return
        # Проверяем размер кеша для реального прогресса
        hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        cache_dir = os.path.join(hf_home, "hub", "models--ZhengPeng7--BiRefNet")
        if os.path.isdir(cache_dir):
            blobs_dir = os.path.join(cache_dir, "blobs")
            if os.path.isdir(blobs_dir):
                try:
                    total_bytes = sum(
                        os.path.getsize(os.path.join(blobs_dir, f))
                        for f in os.listdir(blobs_dir)
                        if not f.startswith(".")
                    )
                    model_size = 424 * 1024 * 1024  # ~424 MB
                    pct = min(95, int(total_bytes / model_size * 100))
                    self._update_progress(pct / 100, f"Downloading model {pct}%")
                except OSError:
                    pass
        self.root.after(500, self._animate_loading)

    # =================================================================
    # Сборка интерфейса
    # =================================================================

    def _build_ui(self):
        """
        Структура UI (сверху вниз):
        ┌─────────────────────────────────────┐
        │  Хедер: заголовок + тег + кнопки    │  row=0
        ├─────────────────────────────────────┤
        │  Контент: две панели с картинками   │  row=1 (растягивается)
        ├─────────────────────────────────────┤
        │  Options row 1: Threshold|Soft edge │  row=2 (после обработки)
        ├─────────────────────────────────────┤
        │  Options row 2: Refine|Erode|Soft α │  row=3 (после обработки)
        ├─────────────────────────────────────┤
        │  Футер: "Powered by BiRefNet"       │  row=4
        └─────────────────────────────────────┘
        """
        # Основной контейнер с отступами 24px от краёв окна
        self.main = ctk.CTkFrame(self.root, fg_color=Colors.BG)
        self.main.pack(fill="both", expand=True, padx=CARD_PADDING, pady=CARD_PADDING)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)  # Контент растягивается

        self._build_header()
        self._build_content()
        self._build_options()
        self._build_footer()

    def _build_header(self):
        """Хедер — белая карточка: [Заголовок + Тег статуса] ... [Open image] [Save image]"""
        self.header = ctk.CTkFrame(
            self.main, fg_color=Colors.CARD, corner_radius=CARD_RADIUS, height=88,
        )
        self.header.grid(row=0, column=0, sticky="ew", pady=(0, CARD_GAP))
        self.header.grid_propagate(False)
        self.header.grid_columnconfigure(1, weight=1)  # Растяжка между левой и правой частью

        # --- Левая часть: заголовок + тег статуса ---
        left = ctk.CTkFrame(self.header, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=(CARD_PADDING, 0), pady=CARD_PADDING)

        # Заголовок "AI Background remover"
        self.title_label = ctk.CTkLabel(
            left, text=APP_NAME, font=self.font_title,
            text_color=Colors.TEXT_PRIMARY, fg_color="transparent",
        )
        self.title_label.pack(side="left")

        # Тег статуса (Loading 0% → Loading 45% → Ready → Done → Image saved / Error)
        self.tag_label = ctk.CTkLabel(
            left, text="Loading 0%", font=self.font_small,
            fg_color=Colors.LOADING_BG, text_color=Colors.LOADING_TEXT,
            corner_radius=TAG_RADIUS, height=24, padx=4, pady=0,
        )
        self.tag_label.pack(side="left", padx=(16, 0))

        # --- Правая часть: кнопки ---
        self.btn_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        self.btn_frame.grid(
            row=0, column=2, sticky="e", padx=(0, CARD_PADDING), pady=CARD_PADDING,
        )

        # Кнопка "Open image" (серая, неактивна пока модель не загружена)
        self.open_btn = ctk.CTkButton(
            self.btn_frame, text="Open image", font=self.font_body,
            fg_color=Colors.SURFACE_NEUTRAL, text_color=Colors.TEXT_SECONDARY,
            hover_color=Colors.SURFACE_NEUTRAL_HOVER,
            corner_radius=BUTTON_RADIUS, height=40, border_width=0,
            command=self._open_image, state="disabled",
        )
        self.open_btn.pack(side="left")

        # Кнопка "Save image" (фиолетовая, скрыта до завершения обработки)
        self.save_btn = ctk.CTkButton(
            self.btn_frame, text="Save image", font=self.font_body,
            fg_color=Colors.PRIMARY, text_color=Colors.PRIMARY_TEXT,
            hover_color=Colors.PRIMARY_HOVER,
            corner_radius=BUTTON_RADIUS, height=40, border_width=0,
            command=self._save_result,
        )

    def _build_options(self):
        """
        Два ряда опций под картинками (появляются после обработки):
        Row 1: [Threshold (grow)] [Soft edge (grow)] [Only mask (fixed)] [Invert (fixed)]
        Row 2: [Refinement (33%)] [Erode (33%)] [Soft alpha (33%)]
        """
        self.options_container = ctk.CTkFrame(self.main, fg_color="transparent")

        # ── Ряд 1: Threshold + Soft edge + Only mask + Invert ─────
        row1 = ctk.CTkFrame(self.options_container, fg_color="transparent")
        row1.pack(fill="x", pady=(0, CARD_GAP))

        # Threshold (grow)
        card_thresh = ctk.CTkFrame(
            row1, fg_color=Colors.CARD, corner_radius=CARD_RADIUS, height=72,
        )
        card_thresh.pack(side="left", fill="x", expand=True, padx=(0, CARD_GAP))
        card_thresh.grid_columnconfigure(1, weight=1)
        card_thresh.grid_propagate(False)

        ctk.CTkLabel(
            card_thresh, text="Threshold", font=self.font_small,
            text_color=Colors.TEXT_SECONDARY, fg_color="transparent",
        ).grid(row=0, column=0, sticky="w", padx=(CARD_PADDING, 16), pady=CARD_PADDING)

        self.slider = CustomSlider(
            card_thresh, from_=0, to=255,
            value=DEFAULT_THRESHOLD, command=self._on_threshold_change,
        )
        self.slider.grid(row=0, column=1, sticky="ew", padx=(0, CARD_PADDING), pady=CARD_PADDING)

        # Soft edge (grow)
        card_fe = ctk.CTkFrame(
            row1, fg_color=Colors.CARD, corner_radius=CARD_RADIUS, height=72,
        )
        card_fe.pack(side="left", fill="x", expand=True, padx=(0, CARD_GAP))
        card_fe.grid_columnconfigure(1, weight=1)
        card_fe.grid_propagate(False)

        ctk.CTkLabel(
            card_fe, text="Soft edge", font=self.font_small,
            text_color=Colors.TEXT_SECONDARY, fg_color="transparent",
        ).grid(row=0, column=0, sticky="w", padx=(CARD_PADDING, 16), pady=CARD_PADDING)

        self.feather_slider = CustomSlider(
            card_fe, from_=0, to=20, value=0,
            command=self._on_feather_change,
        )
        self.feather_slider.grid(row=0, column=1, sticky="ew", padx=(0, CARD_PADDING), pady=CARD_PADDING)

        # Only mask (fixed width, toggle)
        card_mask = ctk.CTkFrame(
            row1, fg_color=Colors.CARD, corner_radius=CARD_RADIUS,
            height=72, width=164,
        )
        card_mask.pack(side="left", padx=(0, CARD_GAP))
        card_mask.pack_propagate(False)

        inner_mask = ctk.CTkFrame(card_mask, fg_color="transparent")
        inner_mask.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            inner_mask, text="Only mask", font=self.font_small,
            text_color=Colors.TEXT_SECONDARY, fg_color="transparent",
        ).pack(side="left", padx=(0, 16))

        self.only_mask_toggle = ToggleSwitch(inner_mask, command=self._on_only_mask_change)
        self.only_mask_toggle.pack(side="left")

        # Invert (fixed width, toggle)
        card_invert = ctk.CTkFrame(
            row1, fg_color=Colors.CARD, corner_radius=CARD_RADIUS,
            height=72, width=137,
        )
        card_invert.pack(side="left")
        card_invert.pack_propagate(False)

        inner_invert = ctk.CTkFrame(card_invert, fg_color="transparent")
        inner_invert.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            inner_invert, text="Invert", font=self.font_small,
            text_color=Colors.TEXT_SECONDARY, fg_color="transparent",
        ).pack(side="left", padx=(0, 16))

        self.invert_toggle = ToggleSwitch(inner_invert, command=self._on_invert_change)
        self.invert_toggle.pack(side="left")

        # ── Ряд 2: Refinement + Erode + Soft alpha ────────────────
        row2 = ctk.CTkFrame(self.options_container, fg_color="transparent")
        row2.pack(fill="x")

        # Refinement (33%)
        card_ref = ctk.CTkFrame(
            row2, fg_color=Colors.CARD, corner_radius=CARD_RADIUS, height=72,
        )
        card_ref.pack(side="left", fill="x", expand=True, padx=(0, CARD_GAP))
        card_ref.grid_columnconfigure(1, weight=1)
        card_ref.grid_propagate(False)

        ctk.CTkLabel(
            card_ref, text="Refinement", font=self.font_small,
            text_color=Colors.TEXT_SECONDARY, fg_color="transparent",
        ).grid(row=0, column=0, sticky="w", padx=(CARD_PADDING, 16), pady=CARD_PADDING)

        self.refine_slider = CustomSlider(
            card_ref, from_=0, to=30, value=0,
            command=self._on_refine_change,
        )
        self.refine_slider.grid(row=0, column=1, sticky="ew", padx=(0, CARD_PADDING), pady=CARD_PADDING)

        # Erode (33%)
        card_erode = ctk.CTkFrame(
            row2, fg_color=Colors.CARD, corner_radius=CARD_RADIUS, height=72,
        )
        card_erode.pack(side="left", fill="x", expand=True, padx=(0, CARD_GAP))
        card_erode.grid_columnconfigure(1, weight=1)
        card_erode.grid_propagate(False)

        ctk.CTkLabel(
            card_erode, text="Erode", font=self.font_small,
            text_color=Colors.TEXT_SECONDARY, fg_color="transparent",
        ).grid(row=0, column=0, sticky="w", padx=(CARD_PADDING, 16), pady=CARD_PADDING)

        self.erode_slider = CustomSlider(
            card_erode, from_=-10, to=10, value=0,
            command=self._on_erode_change,
        )
        self.erode_slider.grid(row=0, column=1, sticky="ew", padx=(0, CARD_PADDING), pady=CARD_PADDING)

        # Soft alpha (33%)
        card_sa = ctk.CTkFrame(
            row2, fg_color=Colors.CARD, corner_radius=CARD_RADIUS, height=72,
        )
        card_sa.pack(side="left", fill="x", expand=True)
        card_sa.grid_columnconfigure(1, weight=1)
        card_sa.grid_propagate(False)

        ctk.CTkLabel(
            card_sa, text="Soft alpha", font=self.font_small,
            text_color=Colors.TEXT_SECONDARY, fg_color="transparent",
        ).grid(row=0, column=0, sticky="w", padx=(CARD_PADDING, 16), pady=CARD_PADDING)

        self.soft_alpha_slider = CustomSlider(
            card_sa, from_=0, to=100, value=0,
            command=self._on_soft_alpha_change,
        )
        self.soft_alpha_slider.grid(row=0, column=1, sticky="ew", padx=(0, CARD_PADDING), pady=CARD_PADDING)

    def _build_content(self):
        """
        Карточка контента — основная область:
        - До загрузки: текст "Drag & drop or open image"
        - После загрузки: две панели рядом [Оригинал | Результат]
        """
        self.content_card = ctk.CTkFrame(
            self.main, fg_color=Colors.CARD, corner_radius=CARD_RADIUS,
        )
        self.content_card.grid(
            row=1, column=0, sticky="nsew", pady=(0, CARD_GAP),
        )

        # Плейсхолдер "Drag & drop or open image" (виден когда нет картинки)
        self.placeholder = ctk.CTkLabel(
            self.content_card, text="Drag & drop or open image",
            font=self.font_title, text_color=Colors.TEXT_PRIMARY,
            fg_color="transparent", cursor="hand2",
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.placeholder.bind("<Button-1>", lambda e: self._open_image())

        # Фрейм с двумя панелями (скрыт пока нет картинки)
        self.panels_frame = ctk.CTkFrame(self.content_card, fg_color="transparent")

        # Левая панель — оригинальное изображение (кликабельна для замены)
        self.original_panel = DashedPanel(self.panels_frame, cursor="hand2")
        self.original_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.original_panel.bind("<Button-1>", lambda e: self._open_image())

        # Правая панель — результат (изображение без фона на шахматке)
        self.result_panel = DashedPanel(self.panels_frame)
        self.result_panel.pack(side="left", fill="both", expand=True, padx=(12, 0))

        # Настройка Drag & Drop
        self._setup_dnd()

    def _build_footer(self):
        """Футер — "Powered by BiRefNet / Stepan Andrushkevich" с кликабельными ссылками."""
        footer_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        footer_frame.grid(row=4, column=0, pady=(8, 0))

        # "Powered by " — обычный текст
        ctk.CTkLabel(
            footer_frame, text="Powered by", font=self.font_small,
            text_color=Colors.TEXT_MUTED, fg_color="transparent",
        ).pack(side="left", padx=(0, 4))

        # "BiRefNet" — ссылка на GitHub автора модели
        birefnet_link = ctk.CTkLabel(
            footer_frame, text="BiRefNet", font=self.font_small,
            text_color=Colors.TEXT_MUTED, fg_color="transparent", cursor="hand2",
        )
        birefnet_link.pack(side="left")
        birefnet_link.bind("<Button-1>", lambda e: self._open_url(
            "https://github.com/ZhengPeng7/BiRefNet"))

        # " / " — разделитель
        ctk.CTkLabel(
            footer_frame, text="/", font=self.font_small,
            text_color=Colors.TEXT_MUTED, fg_color="transparent",
        ).pack(side="left", padx=4)

        # "Stepan Andrushkevich" — ссылка на тг канал
        author_link = ctk.CTkLabel(
            footer_frame, text="Stepan Andrushkevich", font=self.font_small,
            text_color=Colors.TEXT_MUTED, fg_color="transparent", cursor="hand2",
        )
        author_link.pack(side="left")
        author_link.bind("<Button-1>", lambda e: self._open_url(
            "https://t.me/necrondesign"))

    @staticmethod
    def _open_url(url: str):
        """Открывает URL в браузере по умолчанию."""
        import webbrowser
        webbrowser.open(url)

    # =================================================================
    # Drag & Drop (перетаскивание файлов)
    # =================================================================

    def _setup_dnd(self):
        """Подключает tkdnd для приёма файлов перетаскиванием."""
        try:
            tk = self.root.tk

            # Пробуем bundled tkdnd (рядом со скриптом или в PyInstaller _MEIPASS)
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            bundled = os.path.join(base_dir, "libs", "tkdnd")
            if os.path.isdir(bundled):
                tk.eval(f"lappend auto_path {{{bundled}}}")

            # Пробуем tkinterdnd2 из pip / PyInstaller как запасной вариант
            try:
                import tkinterdnd2
                pkg_dir = os.path.join(
                    os.path.dirname(os.path.abspath(tkinterdnd2.__file__)), "tkdnd",
                )
                if os.path.isdir(pkg_dir):
                    tk.eval(f"lappend auto_path {{{pkg_dir}}}")
            except ImportError:
                pass

            # В PyInstaller tkinterdnd2 может лежать в _MEIPASS/tkinterdnd2/tkdnd
            if hasattr(sys, '_MEIPASS'):
                meipass_tkdnd = os.path.join(sys._MEIPASS, "tkinterdnd2", "tkdnd")
                if os.path.isdir(meipass_tkdnd):
                    tk.eval(f"lappend auto_path {{{meipass_tkdnd}}}")

            tk.eval("package require tkdnd")

            # Регистрируем всё окно как drop target
            tk.call("tkdnd::drop_target", "register", ".", "DND_Files")

            # Привязываем обработчик drop-события
            drop_cmd = self.root.register(self._on_dnd_drop)
            tk.eval(f'bind . <<Drop:DND_Files>> {{{drop_cmd} %D}}')

            logger.info("Drag-and-drop enabled (tkdnd)")
        except Exception as exc:
            logger.info("Drag-and-drop not available: %s", exc)

    def _on_dnd_drop(self, data: str):
        """Обработчик: файл перетащили в окно."""
        from urllib.parse import unquote, urlparse

        filepath = data.strip()
        if filepath.startswith("{") and filepath.endswith("}"):
            filepath = filepath[1:-1]  # Windows оборачивает пути в скобки
        if filepath.startswith("file://"):
            filepath = unquote(urlparse(filepath).path)  # декодируем %20, кириллицу и т.д.
        if os.path.isfile(filepath):
            self.root.after(10, lambda p=filepath: self._load_image(p))

    # =================================================================
    # Управление состоянием UI
    # =================================================================

    def _update_progress(self, value: float, text: str = None):
        """Обновляет тег загрузки модели: текст и проценты."""
        if text:
            self.tag_label.configure(
                fg_color=Colors.LOADING_BG, text_color=Colors.LOADING_TEXT, text=text,
            )

    def _update_tag(self, state: str):
        """Обновляет тег статуса (Ready/Done/Error/Image saved)."""
        if state in TAG_STYLES:
            bg, fg, text = TAG_STYLES[state]
            self.tag_label.configure(
                fg_color=bg, text_color=fg, text=text,
            )

    def _show_panels(self):
        """Прячет плейсхолдер и показывает панели с картинками."""
        self.placeholder.place_forget()
        self.panels_frame.place_configure(
            x=CARD_PADDING, y=CARD_PADDING,
            relwidth=1.0, relheight=1.0,
            width=-2 * CARD_PADDING, height=-2 * CARD_PADDING,
        )

    def _show_options(self):
        """Показывает оба ряда опций (после первой обработки)."""
        self.options_container.grid(
            row=2, column=0, sticky="ew", pady=(0, CARD_GAP),
        )

    # =================================================================
    # Загрузка модели
    # =================================================================

    def _load_model(self):
        """Загружает BiRefNet модель в фоновом потоке с прогрессом."""
        try:
            model_path = get_model_path()
            is_local = os.path.isdir(model_path)
            local_only = is_local  # Если модель в кеше — не лезем в интернет
            logger.info("Loading model from %s (local_only=%s)", model_path, local_only)

            if is_local:
                # Модель уже в кеше — показываем прогресс инициализации
                self.root.after(0, lambda: self._update_progress(0.3, "Initializing..."))

            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=local_only,
            )
            self.root.after(0, lambda: self._update_progress(0.7, "Loading to GPU..."))

            self.model.to(self.device)
            self.model.eval()
            self.model.float()

            self.root.after(0, lambda: self._update_progress(1.0, "Ready!"))
            logger.info("Model loaded on device: %s", self.device)
            # Небольшая задержка чтобы пользователь увидел 100%
            time.sleep(0.5)
            self.root.after(0, self._on_model_loaded)
        except Exception as exc:
            logger.exception("Failed to load model")
            self.root.after(0, lambda: self._on_model_error(str(exc)))

    def _on_model_loaded(self):
        """Модель загружена — активируем кнопку Open."""
        self._model_loading = False
        self._update_tag("ready")
        self.open_btn.configure(state="normal")
        logger.info("Model ready, device: %s", self.device.type)

    def _on_model_error(self, message):
        """Ошибка загрузки модели — показываем тег Error."""
        self._model_loading = False
        self._update_tag("error")
        logger.error("Model error: %s", message)

    # =================================================================
    # Загрузка изображения
    # =================================================================

    def _open_image(self):
        """Открывает диалог выбора файла."""
        if self.model is None:
            return

        filepath = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", " ".join(f"*{ext}" for ext in SUPPORTED_FORMATS)),
                ("All files", "*.*"),
            ],
        )
        if filepath:
            self._load_image(filepath)

    def _load_image(self, filepath: str):
        """Загружает изображение и запускает обработку."""
        ext = Path(filepath).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            logger.warning("Unsupported format: %s", ext)
            return

        try:
            self.original_image = Image.open(filepath).convert("RGB")
            self._original_filepath = filepath
        except Exception as exc:
            logger.exception("Failed to open image")
            return

        # Сначала показываем панели, чтобы они получили размеры
        self._show_panels()
        self.root.update_idletasks()

        # Устанавливаем оригинал в левую панель
        self.original_panel.set_image(self.original_image)
        self.result_panel.clear()

        # Запускаем AI-обработку
        self._process_image()

    # =================================================================
    # Инференс (AI-обработка)
    # =================================================================

    def _process_image(self):
        """Запускает инференс модели в фоновом потоке."""
        if self.model is None or self.original_image is None or self._processing:
            return
        self._processing = True
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        """Фоновый поток: прогоняет изображение через BiRefNet → получает маску."""
        try:
            img = self.original_image
            input_tensor = _preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid().cpu()

            mask = preds[0].squeeze()
            mask_pil = transforms.ToPILImage()(mask)
            mask_resized = mask_pil.resize(img.size, Image.BILINEAR)

            self.alpha_matte = np.array(mask_resized, dtype=np.float32)
            logger.info("Inference complete — mask shape %s", self.alpha_matte.shape)
            self.root.after(0, self._on_inference_complete)
        except Exception as exc:
            logger.exception("Inference failed")
        finally:
            self._processing = False

    def _on_inference_complete(self):
        """Обработка завершена — показываем результат, кнопку Save и слайдер."""
        self._update_tag("done")

        # Показываем кнопку Save (один раз)
        if not self._save_btn_shown:
            self.save_btn.pack(side="left", padx=(16, 0))
            self._save_btn_shown = True

        # Показываем опции обработки
        self._show_options()

        # Рендерим превью результата
        self._update_result_preview()

    # =================================================================
    # Threshold и опции обработки маски
    # =================================================================

    def _schedule_preview_update(self):
        """Debounce — обновляет превью с задержкой."""
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(DEBOUNCE_MS, self._update_result_preview)

    def _on_threshold_change(self, value):
        self._threshold = int(value)
        self._schedule_preview_update()

    def _on_feather_change(self, value):
        self._feather = int(value)
        self._schedule_preview_update()

    def _on_erode_change(self, value):
        self._erode_dilate = int(value)
        self._schedule_preview_update()

    def _on_soft_alpha_change(self, value):
        self._soft_alpha = int(value)
        self._schedule_preview_update()

    def _on_refine_change(self, value):
        self._refinement = int(value)
        self._schedule_preview_update()

    def _on_only_mask_change(self, value):
        self._only_mask = bool(value)
        self._schedule_preview_update()

    def _on_invert_change(self, value):
        self._invert = bool(value)
        self._schedule_preview_update()

    def _apply_current_mask(self) -> Image.Image:
        """Применяет маску с текущими параметрами."""
        return apply_mask(
            self.original_image,
            self.alpha_matte,
            threshold=self._threshold,
            soft_alpha=self._soft_alpha,
            erode_dilate=self._erode_dilate,
            feather=self._feather,
            refinement=self._refinement,
            invert=self._invert,
        )

    def _update_result_preview(self):
        """Применяет маску с текущими параметрами и показывает результат."""
        self._debounce_id = None
        if self.original_image is None or self.alpha_matte is None:
            return

        if self._only_mask:
            # Показываем только маску как grayscale
            result = self._apply_current_mask()
            # Извлекаем альфа-канал и показываем как grayscale
            alpha_channel = result.split()[-1]
            self.result_panel.set_image(alpha_channel.convert("RGB"))
        else:
            result = self._apply_current_mask()
            w, h = result.size
            checker = build_checkerboard(w, h).convert("RGBA")
            checker.paste(result, (0, 0), result)
            self.result_panel.set_image(checker.convert("RGB"))

    # =================================================================
    # Экспорт (сохранение)
    # =================================================================

    def _save_result(self):
        """Сохраняет результат (PNG с прозрачностью) через диалог."""
        if self.original_image is None or self.alpha_matte is None:
            return

        original_name = Path(self._original_filepath).stem
        filepath = filedialog.asksaveasfilename(
            title="Save image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=f"{original_name}_no_bg.png",
        )
        if not filepath:
            return

        try:
            result = self._apply_current_mask()
            result.save(filepath, "PNG")
            self._update_tag("saved")
            logger.info("Result saved to %s", filepath)
        except PermissionError:
            logger.error("Cannot save: permission denied for %s", filepath)
        except Exception as exc:
            logger.exception("Save failed")

    # =================================================================
    # Запуск
    # =================================================================

    def run(self):
        """Запускает главный цикл приложения."""
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting %s v%s", APP_NAME, APP_VERSION)
    app = AIBGApp()
    app.run()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
