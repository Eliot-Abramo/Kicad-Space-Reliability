"""Generic themed tabbed page container for wx dialogs."""

from __future__ import annotations

import wx
import wx.lib.buttons as buttons

try:
    from .theme import PALETTE, dip_px, dip_size, style_panel, ui_font
except ImportError:
    from theme import PALETTE, dip_px, dip_size, style_panel, ui_font


class SegmentedBook(wx.Panel):
    """A notebook-like control with fully themed in-app tabs."""

    def __init__(
        self,
        parent,
        *,
        background: wx.Colour | None = None,
        tab_background: wx.Colour | None = None,
        page_background: wx.Colour | None = None,
    ):
        super().__init__(parent)
        self._pages: list[wx.Window] = []
        self._buttons: list[buttons.GenToggleButton] = []
        self._selection = wx.NOT_FOUND
        self._on_page_changed = None

        self._background = background or PALETTE.background
        self._tab_background = tab_background or PALETTE.panel_bg
        self._page_background = page_background or PALETTE.background
        self._active_bg = PALETTE.primary
        self._active_fg = wx.Colour(255, 255, 255)
        self._inactive_bg = PALETTE.card_bg
        self._inactive_fg = PALETTE.text

        style_panel(self, self._background)

        self._tab_bar = wx.Panel(self)
        style_panel(self._tab_bar, self._tab_background)
        self._tab_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._tab_bar.SetSizer(self._tab_sizer)

        self._page_host = wx.Panel(self)
        style_panel(self._page_host, self._page_background)
        self._page_sizer = wx.BoxSizer(wx.VERTICAL)
        self._page_host.SetSizer(self._page_sizer)

        root = wx.BoxSizer(wx.VERTICAL)
        root.Add(self._tab_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, dip_px(self, 8))
        root.Add(self._page_host, 1, wx.EXPAND | wx.ALL, dip_px(self, 8))
        self.SetSizer(root)

    def set_on_page_changed(self, handler) -> None:
        self._on_page_changed = handler

    def AddPage(self, page: wx.Window, text: str, select: bool = False) -> bool:
        if page.GetParent() is not self._page_host:
            page.Reparent(self._page_host)
        page.Hide()
        self._pages.append(page)
        self._page_sizer.Add(page, 1, wx.EXPAND)

        index = len(self._pages) - 1
        button = buttons.GenToggleButton(
            self._tab_bar,
            label=text,
            size=dip_size(self._tab_bar, 120, 32),
        )
        button.SetBezelWidth(1)
        button.SetUseFocusIndicator(False)
        button.SetFont(ui_font(button, role="small", weight=wx.FONTWEIGHT_BOLD))
        button.Bind(wx.EVT_BUTTON, lambda event, idx=index: self.SetSelection(idx))
        self._buttons.append(button)
        self._tab_sizer.Add(button, 0, wx.RIGHT | wx.BOTTOM, dip_px(self._tab_bar, 6))
        self._sync_button_styles()

        if index == 0 or select:
            self.SetSelection(index)
        return True

    def GetPage(self, index: int) -> wx.Window | None:
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def GetPageCount(self) -> int:
        return len(self._pages)

    def GetSelection(self) -> int:
        return self._selection

    def SetSelection(self, index: int) -> int:
        if not 0 <= index < len(self._pages):
            return self._selection

        previous = self._selection
        if previous == index:
            return previous

        if 0 <= previous < len(self._pages):
            self._pages[previous].Hide()

        self._selection = index
        page = self._pages[index]
        page.Show()
        self._sync_button_styles()
        self._page_host.Layout()
        self.Layout()
        page.SendSizeEvent()

        if self._on_page_changed:
            self._on_page_changed()

        return previous

    def _sync_button_styles(self) -> None:
        for index, button in enumerate(self._buttons):
            selected = index == self._selection
            button.SetToggle(selected)
            button.SetBackgroundColour(self._active_bg if selected else self._inactive_bg)
            button.SetForegroundColour(self._active_fg if selected else self._inactive_fg)
            button.SetBestSize(dip_size(button, 120, 32))
            button.Refresh()

