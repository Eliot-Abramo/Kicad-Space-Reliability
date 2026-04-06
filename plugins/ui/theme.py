"""
Adaptive theme helpers for KiCad wx dialogs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import wx


IS_WINDOWS = sys.platform.startswith("win")
WINDOWS_FONT_POINT_DELTA = -3


def _colour_to_rgb(colour: wx.Colour) -> tuple[int, int, int]:
    return int(colour.Red()), int(colour.Green()), int(colour.Blue())


def _luminance(colour: wx.Colour) -> float:
    def channel(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = _colour_to_rgb(colour)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast_ratio(fg: wx.Colour, bg: wx.Colour) -> float:
    l1 = _luminance(fg)
    l2 = _luminance(bg)
    hi = max(l1, l2)
    lo = min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _pick_text(bg: wx.Colour, light: wx.Colour, dark: wx.Colour) -> wx.Colour:
    return light if _contrast_ratio(light, bg) >= _contrast_ratio(dark, bg) else dark


def _safe_sys_colour(which: int, fallback: wx.Colour) -> wx.Colour:
    try:
        colour = wx.SystemSettings.GetColour(which)
        if colour.IsOk():
            return colour
    except Exception:
        pass
    return fallback


def _system_prefers_dark() -> bool:
    try:
        appearance = wx.SystemSettings.GetAppearance()
        return bool(appearance and appearance.IsDark())
    except Exception:
        return False


def _explicit_theme_mode() -> str | None:
    mode = os.environ.get("KICAD_RELIABILITY_THEME")
    if mode is None:
        return None
    mode = mode.strip().lower()
    if mode in {"dark", "light", "system"}:
        return mode
    return None


def _theme_mode() -> str:
    mode = _explicit_theme_mode()
    if mode:
        return mode
    if IS_WINDOWS:
        return "dark"
    return "dark"


def platform_point_size(size: int, minimum: int = 8) -> int:
    delta = WINDOWS_FONT_POINT_DELTA if IS_WINDOWS else 0
    return max(minimum, size + delta)


def _base_font_for(window: wx.Window | None = None) -> wx.Font:
    candidates = []
    if window is not None:
        try:
            candidates.append(window.GetFont())
        except Exception:
            pass
    try:
        candidates.append(wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT))
    except Exception:
        pass
    try:
        candidates.append(wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT))
    except Exception:
        pass
    for font in candidates:
        if font and font.IsOk():
            return wx.Font(font)
    return wx.Font(wx.NORMAL_FONT)


def tuned_font(
    window: wx.Window | None = None,
    *,
    relative: int = 0,
    family: int | None = None,
    style: int | None = None,
    weight: int | None = None,
    minimum: int = 8,
) -> wx.Font:
    font = _base_font_for(window)
    font.SetPointSize(max(minimum, font.GetPointSize() + relative + (WINDOWS_FONT_POINT_DELTA if IS_WINDOWS else 0)))
    if family is not None:
        font.SetFamily(family)
    if style is not None:
        font.SetStyle(style)
    if weight is not None:
        font.SetWeight(weight)
    return font


def ui_font(
    window: wx.Window | None = None,
    *,
    role: str = "body",
    relative: int = 0,
    family: int | None = None,
    style: int | None = None,
    weight: int | None = None,
    minimum: int = 8,
) -> wx.Font:
    role_relative = {
        "caption": -2,
        "small": -1,
        "body": 0,
        "section": 1,
        "title": 2,
        "hero": 4,
        "mono": -1,
    }.get(role, 0)
    if role == "mono" and family is None:
        family = wx.FONTFAMILY_TELETYPE
    return tuned_font(
        window,
        relative=role_relative + relative,
        family=family,
        style=style,
        weight=weight,
        minimum=minimum,
    )


def apply_compact_fonts(window: wx.Window | None, delta: int | None = None, minimum: int = 8) -> None:
    if not window or not IS_WINDOWS:
        return

    delta = WINDOWS_FONT_POINT_DELTA if delta is None else delta

    def _shrink(node: wx.Window) -> None:
        try:
            font = node.GetFont()
            if font and font.IsOk():
                sized = wx.Font(font)
                sized.SetPointSize(max(minimum, font.GetPointSize() + delta))
                node.SetFont(sized)
        except Exception:
            pass

        try:
            for child in node.GetChildren():
                _shrink(child)
        except Exception:
            pass

    _shrink(window)


def dip_px(window: wx.Window, value: int) -> int:
    if value < 0:
        return value
    try:
        return int(window.FromDIP(value))
    except Exception:
        return value


def dip_size(window: wx.Window, width: int, height: int = -1) -> wx.Size:
    return wx.Size(dip_px(window, width), dip_px(window, height))


def _same_colour(lhs: wx.Colour, rhs: wx.Colour, tolerance: int = 10) -> bool:
    if not lhs or not rhs or not lhs.IsOk() or not rhs.IsOk():
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(_colour_to_rgb(lhs), _colour_to_rgb(rhs)))


def _is_native_light_surface(colour: wx.Colour) -> bool:
    if not colour or not colour.IsOk():
        return True
    if _luminance(colour) < 0.55:
        return False
    for sys_colour in (
        _safe_sys_colour(wx.SYS_COLOUR_WINDOW, wx.Colour(255, 255, 255)),
        _safe_sys_colour(wx.SYS_COLOUR_3DFACE, wx.Colour(245, 246, 247)),
        _safe_sys_colour(wx.SYS_COLOUR_BTNFACE, wx.Colour(240, 240, 240)),
    ):
        if _same_colour(colour, sys_colour):
            return True
    return _luminance(colour) > 0.80


def _set_surface(window: wx.Window, background: wx.Colour | None = None, foreground: wx.Colour | None = None) -> None:
    if foreground is not None:
        try:
            window.SetForegroundColour(foreground)
        except Exception:
            pass
    if background is not None:
        try:
            window.SetBackgroundColour(background)
        except Exception:
            pass
        try:
            window.SetOwnBackgroundColour(background)
        except Exception:
            pass


@dataclass(frozen=True)
class Palette:
    background: wx.Colour
    panel_bg: wx.Colour
    card_bg: wx.Colour
    field_bg: wx.Colour
    header_bg: wx.Colour
    header_fg: wx.Colour
    border: wx.Colour
    grid: wx.Colour
    text: wx.Colour
    text_muted: wx.Colour
    text_soft: wx.Colour
    accent: wx.Colour
    primary: wx.Colour
    success: wx.Colour
    warning: wx.Colour
    danger: wx.Colour
    info_bg: wx.Colour
    row_alt: wx.Colour
    is_dark: bool


def build_palette() -> Palette:
    sys_window = _safe_sys_colour(wx.SYS_COLOUR_WINDOW, wx.Colour(255, 255, 255))
    sys_face = _safe_sys_colour(wx.SYS_COLOUR_3DFACE, wx.Colour(245, 246, 247))
    sys_text = _safe_sys_colour(wx.SYS_COLOUR_WINDOWTEXT, wx.Colour(32, 33, 36))
    sys_button = _safe_sys_colour(wx.SYS_COLOUR_BTNFACE, wx.Colour(240, 240, 240))
    theme_mode = _theme_mode()
    prefers_dark = True if theme_mode == "dark" else _system_prefers_dark()
    if theme_mode == "light":
        prefers_dark = False
    dark_like = prefers_dark or _luminance(sys_window) < 0.35 or _contrast_ratio(sys_text, sys_window) < 4.5

    if dark_like:
        background = wx.Colour(31, 36, 46)
        panel_bg = wx.Colour(36, 42, 54)
        card_bg = wx.Colour(43, 50, 64)
        field_bg = wx.Colour(48, 57, 73)
        header_bg = wx.Colour(21, 27, 38)
        border = wx.Colour(88, 101, 125)
        grid = wx.Colour(73, 86, 108)
        text = wx.Colour(238, 243, 249)
        text_muted = wx.Colour(192, 201, 214)
        text_soft = wx.Colour(152, 164, 181)
        primary = wx.Colour(91, 196, 255)
        accent = wx.Colour(53, 215, 173)
        success = wx.Colour(52, 199, 132)
        warning = wx.Colour(245, 177, 66)
        danger = wx.Colour(239, 104, 104)
        info_bg = wx.Colour(37, 57, 91)
        row_alt = wx.Colour(39, 46, 59)
    else:
        background = wx.Colour(241, 245, 249)
        panel_bg = wx.Colour(248, 250, 252)
        card_bg = wx.Colour(255, 255, 255)
        field_bg = wx.Colour(246, 248, 251)
        header_bg = wx.Colour(24, 48, 84)
        border = wx.Colour(209, 219, 231)
        grid = wx.Colour(226, 233, 241)
        text = _pick_text(card_bg, wx.Colour(255, 255, 255), wx.Colour(27, 34, 45))
        text_muted = wx.Colour(88, 102, 122)
        text_soft = wx.Colour(120, 133, 150)
        primary = wx.Colour(36, 99, 190)
        accent = wx.Colour(16, 126, 103)
        success = wx.Colour(24, 136, 96)
        warning = wx.Colour(183, 129, 33)
        danger = wx.Colour(187, 64, 69)
        info_bg = wx.Colour(232, 240, 251)
        row_alt = wx.Colour(243, 246, 250)

    header_fg = _pick_text(header_bg, wx.Colour(255, 255, 255), wx.Colour(17, 24, 39))
    if _contrast_ratio(text, card_bg) < 5.0:
        text = _pick_text(card_bg, wx.Colour(255, 255, 255), wx.Colour(22, 28, 37))
    if _contrast_ratio(text_muted, card_bg) < 3.5:
        text_muted = _pick_text(card_bg, wx.Colour(228, 232, 238), wx.Colour(84, 96, 112))

    return Palette(
        background=background,
        panel_bg=panel_bg,
        card_bg=card_bg,
        field_bg=field_bg,
        header_bg=header_bg,
        header_fg=header_fg,
        border=border,
        grid=grid,
        text=text,
        text_muted=text_muted,
        text_soft=text_soft,
        accent=accent,
        primary=primary,
        success=success,
        warning=warning,
        danger=danger,
        info_bg=info_bg,
        row_alt=row_alt,
        is_dark=dark_like,
    )


PALETTE = build_palette()


def style_panel(panel: wx.Window, background: wx.Colour | None = None) -> None:
    _set_surface(panel, background or PALETTE.card_bg, PALETTE.text)


def style_text_like(ctrl: wx.Window, read_only: bool = False) -> None:
    _set_surface(ctrl, PALETTE.field_bg if read_only else PALETTE.card_bg, PALETTE.text)


def style_list_ctrl(ctrl: wx.ListCtrl) -> None:
    _set_surface(ctrl, PALETTE.card_bg, PALETTE.text)


def apply_theme_recursively(
    window: wx.Window | None,
    *,
    background: wx.Colour | None = None,
    force_background: bool = False,
) -> None:
    if not window:
        return

    def _style(node: wx.Window, inherited_bg: wx.Colour | None, force_bg: bool) -> None:
        next_bg = inherited_bg or PALETTE.panel_bg

        if isinstance(node, wx.Dialog):
            next_bg = inherited_bg or PALETTE.background
            if force_bg or _is_native_light_surface(node.GetBackgroundColour()):
                _set_surface(node, next_bg, PALETTE.text)
        elif isinstance(node, wx.Notebook):
            next_bg = inherited_bg or PALETTE.background
            _set_surface(node, next_bg, PALETTE.text)
        elif isinstance(node, wx.SplitterWindow):
            splitter_bg = PALETTE.border if PALETTE.is_dark else PALETTE.grid
            _set_surface(node, splitter_bg, PALETTE.text)
            try:
                node.SetBorderSize(0)
            except Exception:
                pass
            try:
                node.SetSashSize(dip_px(node, 6))
            except Exception:
                pass
        elif isinstance(node, (wx.Panel, wx.ScrolledWindow)):
            if force_bg or _is_native_light_surface(node.GetBackgroundColour()):
                _set_surface(node, next_bg, PALETTE.text)
        elif isinstance(node, wx.TextCtrl):
            read_only = bool(node.GetWindowStyleFlag() & wx.TE_READONLY)
            style_text_like(node, read_only=read_only)
            next_bg = PALETTE.field_bg if read_only else PALETTE.card_bg
        elif isinstance(node, wx.ListCtrl):
            style_list_ctrl(node)
            next_bg = PALETTE.card_bg
        elif isinstance(node, wx.ListBox):
            _set_surface(node, PALETTE.card_bg, PALETTE.text)
            next_bg = PALETTE.card_bg
        elif isinstance(node, (wx.ComboBox, wx.Choice, wx.SpinCtrl, wx.SpinCtrlDouble)):
            _set_surface(node, PALETTE.card_bg, PALETTE.text)
            next_bg = PALETTE.card_bg
        elif isinstance(node, wx.StaticBox):
            _set_surface(node, foreground=PALETTE.text)
        elif isinstance(node, wx.StaticText):
            _set_surface(node, foreground=PALETTE.text)

        try:
            children = list(node.GetChildren())
        except Exception:
            children = []

        for child in children:
            child_bg = next_bg
            if isinstance(node, wx.Notebook):
                child_bg = PALETTE.background
            _style(child, child_bg, force_bg)

    _style(window, background, force_background)
