"""Smoke tests for wx-dependent modules — requires display (xvfb in CI).

Marked with pytest.mark.wx to enable selective CI execution.
"""

import pytest

pytestmark = pytest.mark.wx


def test_theme_module_imports():
    import wx

    from plugins.ui.theme import Palette, build_palette

    app = wx.App(False)
    palette = build_palette()
    assert isinstance(palette, Palette)
    app.Destroy()


def test_windowing_module_imports():
    import wx

    from plugins.ui.windowing import center_dialog, get_display_client_area

    app = wx.App(False)
    # Just verify the functions are callable
    assert callable(get_display_client_area)
    assert callable(center_dialog)
    app.Destroy()


def test_book_module_imports():
    import wx

    from plugins.ui.book import SegmentedBook

    app = wx.App(False)
    frame = wx.Frame(None)
    book = SegmentedBook(frame)
    assert book is not None
    frame.Destroy()
    app.Destroy()


def test_theme_palette_values():
    import wx

    from plugins.ui.theme import build_palette

    app = wx.App(False)
    palette = build_palette()
    assert isinstance(palette.background, wx.Colour)
    assert isinstance(palette.text, wx.Colour)
    assert isinstance(palette.is_dark, bool)
    app.Destroy()
