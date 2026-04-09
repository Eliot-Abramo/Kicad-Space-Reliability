"""
Window placement helpers for wx dialogs.
Author:  Eliot Abramo
"""

import wx


def _shown_on_screen(window) -> bool:
    try:
        return bool(window and window.IsShownOnScreen())
    except Exception:
        return False


def get_display_client_area(window=None, parent=None):
    """Return the best display client area for the given window context."""
    display_index = wx.NOT_FOUND

    for candidate in (parent, window):
        try:
            if candidate:
                display_index = wx.Display.GetFromWindow(candidate)
                if display_index != wx.NOT_FOUND:
                    break
        except Exception:
            pass

    if display_index == wx.NOT_FOUND and _shown_on_screen(parent):
        try:
            rect = parent.GetScreenRect()
            center = wx.Point(rect.x + rect.width // 2, rect.y + rect.height // 2)
            display_index = wx.Display.GetFromPoint(center)
        except Exception:
            pass

    if display_index == wx.NOT_FOUND:
        display_index = 0

    try:
        return wx.Display(display_index).GetClientArea()
    except Exception:
        return wx.Display(0).GetClientArea()


def center_dialog(window, parent=None):
    """Center a dialog on its parent or best matching display client area."""
    if not window:
        return

    try:
        window.Layout()
    except Exception:
        pass

    parent = parent or window.GetParent()
    display_rect = get_display_client_area(window, parent)
    width, height = window.GetSize()

    if _shown_on_screen(parent):
        try:
            parent_rect = parent.GetScreenRect()
            x = parent_rect.x + max(0, (parent_rect.width - width) // 2)
            y = parent_rect.y + max(0, (parent_rect.height - height) // 2)
        except Exception:
            x = display_rect.x + max(0, (display_rect.width - width) // 2)
            y = display_rect.y + max(0, (display_rect.height - height) // 2)
    else:
        x = display_rect.x + max(0, (display_rect.width - width) // 2)
        y = display_rect.y + max(0, (display_rect.height - height) // 2)

    max_x = display_rect.x + max(0, display_rect.width - width)
    max_y = display_rect.y + max(0, display_rect.height - height)
    x = min(max(x, display_rect.x), max_x)
    y = min(max(y, display_rect.y), max_y)

    try:
        window.SetPosition((x, y))
    except Exception:
        try:
            if _shown_on_screen(parent):
                window.CentreOnParent()
            else:
                window.CentreOnScreen()
        except Exception:
            window.Centre()
