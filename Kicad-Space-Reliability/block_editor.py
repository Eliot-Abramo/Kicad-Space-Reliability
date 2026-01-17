"""
Visual Block Diagram Editor

Drag-and-drop canvas for defining reliability topology.
"""

import wx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .reliability_math import ConnectionType


@dataclass
class Block:
    """A block in the reliability diagram."""
    id: str
    name: str  # Full sheet path
    label: str  # Display name
    x: int = 0
    y: int = 0
    width: int = 150
    height: int = 55
    reliability: float = 1.0
    lambda_val: float = 0.0
    is_group: bool = False
    children: List[str] = field(default_factory=list)
    connection_type: str = "series"
    k_value: int = 2

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class BlockEditor(wx.Panel):
    """Visual editor for reliability block diagrams with zoom and pan support."""

    GRID = 20
    BLOCK_W = 180  # Wider blocks for better readability
    BLOCK_H = 70   # Taller blocks
    PAD = 20       # More padding around groups

    # Colors
    BG = wx.Colour(248, 248, 248)
    GRID_COLOR = wx.Colour(230, 230, 230)
    BLOCK_COLOR = wx.Colour(200, 220, 255)
    BLOCK_SEL = wx.Colour(150, 180, 255)
    SERIES_COLOR = wx.Colour(220, 255, 220)
    PARALLEL_COLOR = wx.Colour(255, 220, 220)
    KN_COLOR = wx.Colour(255, 255, 200)

    def __init__(self, parent):
        super().__init__(parent, style=wx.BORDER_SIMPLE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetMinSize((900, 500))

        self.blocks: Dict[str, Block] = {}
        self.root_id: Optional[str] = None
        self.selected: Optional[str] = None
        self.hover: Optional[str] = None

        self.dragging = False
        self.drag_offset = (0, 0)
        self.selecting = False
        self.sel_start = (0, 0)
        self.sel_rect: Optional[Tuple[int, int, int, int]] = None

        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.panning = False
        self.pan_start = (0, 0)

        self.mission_hours = 5 * 365 * 24

        # Callbacks
        self.on_selection_change = None
        self.on_structure_change = None
        self.on_block_activate = None
        self.on_zoom_change = None

        # Events
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        self.Bind(wx.EVT_LEFT_DCLICK, self._on_dclick)
        self.Bind(wx.EVT_RIGHT_DOWN, self._on_right_click)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_mouse_wheel)
        self.Bind(wx.EVT_MIDDLE_DOWN, self._on_middle_down)
        self.Bind(wx.EVT_MIDDLE_UP, self._on_middle_up)
        self.Bind(wx.EVT_KEY_DOWN, self._on_key)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

        self.SetFocus()

    # === Zoom/Pan methods ===

    def screen_to_canvas(self, sx, sy):
        """Convert screen coordinates to canvas coordinates."""
        cx = (sx - self.pan_offset[0]) / self.zoom_level
        cy = (sy - self.pan_offset[1]) / self.zoom_level
        return int(cx), int(cy)

    def canvas_to_screen(self, cx, cy):
        """Convert canvas coordinates to screen coordinates."""
        sx = cx * self.zoom_level + self.pan_offset[0]
        sy = cy * self.zoom_level + self.pan_offset[1]
        return int(sx), int(sy)

    def set_zoom(self, zoom, center_on=None):
        """Set zoom level, optionally centered on a screen point."""
        old_zoom = self.zoom_level
        new_zoom = max(0.25, min(3.0, zoom))

        if abs(new_zoom - old_zoom) < 0.01:
            return

        if center_on:
            cx, cy = center_on
            # Get canvas point under cursor
            canvas_x = (cx - self.pan_offset[0]) / old_zoom
            canvas_y = (cy - self.pan_offset[1]) / old_zoom
            # Adjust pan to keep that point under cursor
            self.pan_offset[0] = cx - canvas_x * new_zoom
            self.pan_offset[1] = cy - canvas_y * new_zoom

        self.zoom_level = new_zoom
        self.Refresh()

        if self.on_zoom_change:
            self.on_zoom_change(new_zoom)

    def zoom_in(self, center_on=None):
        self.set_zoom(self.zoom_level + 0.1, center_on)

    def zoom_out(self, center_on=None):
        self.set_zoom(self.zoom_level - 0.1, center_on)

    def zoom_fit(self):
        """Fit all blocks in view."""
        if not self.blocks:
            self.zoom_reset()
            return

        # Find bounding box of all blocks
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for b in self.blocks.values():
            min_x = min(min_x, b.x)
            min_y = min(min_y, b.y)
            max_x = max(max_x, b.x + b.width)
            max_y = max(max_y, b.y + b.height)

        if min_x == float('inf'):
            return

        # Add padding
        pad = 40
        content_w = max_x - min_x + pad * 2
        content_h = max_y - min_y + pad * 2

        # Calculate zoom to fit
        panel_w, panel_h = self.GetSize()
        if panel_w <= 0 or panel_h <= 0:
            return

        zoom_x = panel_w / content_w if content_w > 0 else 1.0
        zoom_y = panel_h / content_h if content_h > 0 else 1.0
        fit_zoom = min(zoom_x, zoom_y, 2.0)
        fit_zoom = max(fit_zoom, 0.25)

        self.zoom_level = fit_zoom

        # Center content
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        self.pan_offset[0] = panel_w / 2 - center_x * fit_zoom
        self.pan_offset[1] = panel_h / 2 - center_y * fit_zoom

        self.Refresh()
        if self.on_zoom_change:
            self.on_zoom_change(fit_zoom)

    def zoom_reset(self):
        """Reset zoom to 100% and pan to origin."""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.Refresh()
        if self.on_zoom_change:
            self.on_zoom_change(1.0)

    def get_zoom_percent(self):
        return int(round(self.zoom_level * 100))

    # === Block management ===

    def add_block(self, block_id: str, name: str, label: str = None) -> Block:
        """Add a sheet block."""
        if label is None:
            label = name.rstrip('/').split('/')[-1] or name

        x, y = self._find_position()

        block = Block(id=block_id, name=name, label=label, x=x, y=y,
                      width=self.BLOCK_W, height=self.BLOCK_H)
        self.blocks[block_id] = block

        # Create root group if needed
        if self.root_id is None:
            root = Block(id="__root__", name="System", label="System",
                        is_group=True, connection_type="series")
            self.blocks["__root__"] = root
            self.root_id = "__root__"

        # Add to root
        root = self.blocks.get(self.root_id)
        if root and block_id not in root.children:
            root.children.append(block_id)

        self._update_group_bounds()
        self.Refresh()
        return block

    def remove_block(self, block_id: str):
        """Remove a block."""
        if block_id not in self.blocks:
            return

        # Remove from parents
        for b in self.blocks.values():
            if b.is_group and block_id in b.children:
                b.children.remove(block_id)

        del self.blocks[block_id]
        if self.selected == block_id:
            self.selected = None

        self._update_group_bounds()
        self.Refresh()
        self._notify_change()

    def create_group(self, block_ids: List[str], conn_type: str, k: int = 2) -> Optional[str]:
        """Group blocks together."""
        if len(block_ids) < 2:
            return None

        gid = f"__grp_{sum(1 for b in self.blocks.values() if b.is_group)}__"

        # Bounds
        min_x = min(self.blocks[bid].x for bid in block_ids)
        min_y = min(self.blocks[bid].y for bid in block_ids)
        max_x = max(self.blocks[bid].x + self.blocks[bid].width for bid in block_ids)
        max_y = max(self.blocks[bid].y + self.blocks[bid].height for bid in block_ids)

        label = {
            "series": "SERIES",
            "parallel": "PARALLEL",
            "k_of_n": f"{k}-of-{len(block_ids)}"
        }[conn_type]

        group = Block(
            id=gid, name=label, label=label,
            x=min_x - self.PAD, y=min_y - self.PAD,
            width=max_x - min_x + 2*self.PAD,
            height=max_y - min_y + 2*self.PAD,
            is_group=True, children=list(block_ids),
            connection_type=conn_type, k_value=k
        )
        self.blocks[gid] = group

        # Move blocks from root to this group
        root = self.blocks.get(self.root_id)
        if root:
            for bid in block_ids:
                if bid in root.children:
                    root.children.remove(bid)
            root.children.append(gid)

        self.Refresh()
        self._notify_change()
        return gid

    def ungroup(self, group_id: str):
        """Dissolve a group."""
        if group_id not in self.blocks or not self.blocks[group_id].is_group:
            return

        group = self.blocks[group_id]
        children = group.children.copy()

        # Find parent
        parent = None
        for b in self.blocks.values():
            if b.is_group and group_id in b.children:
                parent = b
                break

        if parent:
            parent.children.remove(group_id)
            parent.children.extend(children)

        del self.blocks[group_id]
        self._update_group_bounds()
        self.Refresh()
        self._notify_change()

    def _find_position(self) -> Tuple[int, int]:
        """Find free position for new block with good spacing."""
        x, y = self.PAD + self.GRID, self.PAD + self.GRID
        spacing_x = self.BLOCK_W + 40  # Horizontal gap between blocks
        spacing_y = self.BLOCK_H + 40  # Vertical gap between blocks
        max_x = 800  # Max x before wrapping to next row

        while True:
            collision = False
            for b in self.blocks.values():
                if not b.is_group:
                    if abs(b.x - x) < spacing_x and abs(b.y - y) < spacing_y:
                        collision = True
                        break

            if not collision:
                return (x, y)

            x += spacing_x
            if x > max_x:
                x = self.PAD + self.GRID
                y += spacing_y

    def _snap(self, x: int, y: int) -> Tuple[int, int]:
        return (round(x / self.GRID) * self.GRID, round(y / self.GRID) * self.GRID)

    def _block_at(self, x: int, y: int) -> Optional[str]:
        """Get block at canvas position (prefer non-groups)."""
        for bid, b in self.blocks.items():
            if not b.is_group and b.contains(x, y):
                return bid
        for bid, b in self.blocks.items():
            if b.is_group and b.contains(x, y):
                return bid
        return None

    def _update_group_bounds(self):
        """Update group boundaries."""
        for g in self.blocks.values():
            if g.is_group and g.children:
                min_x = min_y = float('inf')
                max_x = max_y = float('-inf')

                for cid in g.children:
                    c = self.blocks.get(cid)
                    if c:
                        min_x = min(min_x, c.x)
                        min_y = min(min_y, c.y)
                        max_x = max(max_x, c.x + c.width)
                        max_y = max(max_y, c.y + c.height)

                if min_x != float('inf'):
                    g.x = int(min_x - self.PAD)
                    g.y = int(min_y - self.PAD)
                    g.width = int(max_x - min_x + 2*self.PAD)
                    g.height = int(max_y - min_y + 2*self.PAD)

    def _notify_change(self):
        if self.on_structure_change:
            self.on_structure_change()

    def _notify_selection(self):
        if self.on_selection_change:
            self.on_selection_change(self.selected)

    # === Event handlers ===

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        w, h = self.GetSize()

        # Background
        gc.SetBrush(wx.Brush(self.BG))
        gc.DrawRectangle(0, 0, w, h)

        # Apply zoom and pan
        gc.Translate(self.pan_offset[0], self.pan_offset[1])
        gc.Scale(self.zoom_level, self.zoom_level)

        # Draw grid
        gc.SetPen(wx.Pen(self.GRID_COLOR, 1))
        # Calculate visible area in canvas coords
        vis_x0, vis_y0 = self.screen_to_canvas(0, 0)
        vis_x1, vis_y1 = self.screen_to_canvas(w, h)

        # Draw grid lines within visible area
        start_x = (vis_x0 // self.GRID) * self.GRID
        start_y = (vis_y0 // self.GRID) * self.GRID
        for x in range(start_x, vis_x1 + self.GRID, self.GRID):
            gc.StrokeLine(x, vis_y0, x, vis_y1)
        for y in range(start_y, vis_y1 + self.GRID, self.GRID):
            gc.StrokeLine(vis_x0, y, vis_x1, y)

        # Groups (back)
        for b in sorted(self.blocks.values(), key=lambda x: not x.is_group):
            if b.is_group:
                self._draw_group(gc, b)

        # Blocks (front)
        for b in self.blocks.values():
            if not b.is_group:
                self._draw_block(gc, b)

        # Reset transform for overlays
        gc.SetTransform(gc.CreateMatrix())

        # Selection rectangle (screen coords)
        if self.selecting and self.sel_rect:
            x, y, sw, sh = self.sel_rect
            gc.SetBrush(wx.Brush(wx.Colour(100, 150, 255, 50)))
            gc.SetPen(wx.Pen(wx.Colour(100, 150, 255), 2, wx.PENSTYLE_DOT))
            gc.DrawRectangle(x, y, sw, sh)

        # Zoom indicator
        zoom_text = f"{self.get_zoom_percent()}%"
        font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font, wx.Colour(100, 100, 100))
        tw, th = gc.GetTextExtent(zoom_text)[:2]
        gc.SetBrush(wx.Brush(wx.Colour(255, 255, 255)))
        gc.SetPen(wx.Pen(wx.Colour(200, 200, 200), 1))
        gc.DrawRoundedRectangle(w - tw - 12, h - th - 10, tw + 8, th + 6, 3)
        gc.DrawText(zoom_text, w - tw - 8, h - th - 7)

    def _draw_block(self, gc, b: Block):
        if b.id == self.selected:
            gc.SetBrush(wx.Brush(self.BLOCK_SEL))
            gc.SetPen(wx.Pen(wx.Colour(50, 100, 200), 3))
        elif b.id == self.hover:
            gc.SetBrush(wx.Brush(self.BLOCK_COLOR))
            gc.SetPen(wx.Pen(wx.Colour(100, 150, 200), 2))
        else:
            gc.SetBrush(wx.Brush(self.BLOCK_COLOR))
            gc.SetPen(wx.Pen(wx.Colour(100, 100, 100), 1))

        gc.DrawRoundedRectangle(b.x, b.y, b.width, b.height, 8)

        # Label - larger font, better positioned
        font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        gc.SetFont(font, wx.Colour(30, 30, 30))

        label = b.label[:20] + "..." if len(b.label) > 20 else b.label
        tw = gc.GetTextExtent(label)[0]
        gc.DrawText(label, b.x + (b.width - tw)/2, b.y + 10)

        # Reliability - better spacing
        font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font, wx.Colour(60, 60, 60))

        r_text = f"R = {b.reliability:.4f}"
        tw = gc.GetTextExtent(r_text)[0]
        gc.DrawText(r_text, b.x + (b.width - tw)/2, b.y + 30)

        l_text = f"λ = {b.lambda_val:.2e}"
        tw = gc.GetTextExtent(l_text)[0]
        gc.DrawText(l_text, b.x + (b.width - tw)/2, b.y + 48)

    def _draw_group(self, gc, g: Block):
        color = {
            "series": self.SERIES_COLOR,
            "parallel": self.PARALLEL_COLOR,
            "k_of_n": self.KN_COLOR,
        }.get(g.connection_type, self.SERIES_COLOR)

        gc.SetBrush(wx.Brush(wx.Colour(color.Red(), color.Green(), color.Blue(), 80)))

        if g.id == self.selected:
            gc.SetPen(wx.Pen(wx.Colour(50, 100, 200), 3, wx.PENSTYLE_DOT))
        else:
            gc.SetPen(wx.Pen(wx.Colour(150, 150, 150), 2, wx.PENSTYLE_DOT))

        gc.DrawRoundedRectangle(g.x, g.y, g.width, g.height, 10)

        # Label
        font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        gc.SetFont(font, wx.Colour(80, 80, 80))

        label = g.label
        gc.DrawText(label, g.x + 6, g.y + 3)

        r_text = f"R={g.reliability:.4f}"
        tw = gc.GetTextExtent(r_text)[0]
        gc.DrawText(r_text, g.x + g.width - tw - 6, g.y + 3)

    def _on_left_down(self, event):
        sx, sy = event.GetPosition()
        cx, cy = self.screen_to_canvas(sx, sy)
        self.SetFocus()

        bid = self._block_at(cx, cy)

        if bid:
            # Click on a block - select and start dragging
            self.selected = bid
            self.dragging = True
            b = self.blocks[bid]
            self.drag_offset = (cx - b.x, cy - b.y)
            self._notify_selection()
        elif event.ShiftDown():
            # Shift+click on empty space - start selection rectangle
            self.selecting = True
            self.sel_start = (sx, sy)
            self.sel_rect = (sx, sy, 0, 0)
            self.selected = None
            self._notify_selection()
        else:
            # Click on empty space - start panning
            self.panning = True
            self.pan_start = (sx, sy)
            self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
            self.selected = None
            self._notify_selection()

        self.Refresh()

    def _on_left_up(self, event):
        if self.dragging:
            self.dragging = False
            self._update_group_bounds()
            self._notify_change()

        if self.panning:
            self.panning = False
            self.SetCursor(wx.Cursor(wx.CURSOR_DEFAULT))

        if self.selecting:
            self.selecting = False
            if self.sel_rect:
                sx, sy, sw, sh = self.sel_rect
                if sw > 20 and sh > 20:
                    selected = []
                    for bid, b in self.blocks.items():
                        if not b.is_group:
                            bcx, bcy = b.center()
                            scx, scy = self.canvas_to_screen(bcx, bcy)
                            if sx <= scx <= sx + sw and sy <= scy <= sy + sh:
                                selected.append(bid)

                    if len(selected) >= 2:
                        self._show_group_menu(selected)

            self.sel_rect = None

        self.Refresh()

    def _on_dclick(self, event):
        sx, sy = event.GetPosition()
        cx, cy = self.screen_to_canvas(sx, sy)
        bid = self._block_at(cx, cy)

        if not bid:
            return

        b = self.blocks[bid]
        if b.is_group:
            self._edit_group(bid)
        elif self.on_block_activate:
            self.on_block_activate(bid, b.name)

    def _on_right_click(self, event):
        sx, sy = event.GetPosition()
        cx, cy = self.screen_to_canvas(sx, sy)
        bid = self._block_at(cx, cy)

        if bid:
            self.selected = bid
            self._notify_selection()
            self.Refresh()

            menu = wx.Menu()
            b = self.blocks[bid]

            if b.is_group:
                item = menu.Append(wx.ID_ANY, "Edit Group Type...")
                self.Bind(wx.EVT_MENU, lambda e: self._edit_group(bid), item)
                menu.AppendSeparator()
                item = menu.Append(wx.ID_ANY, "Ungroup")
                self.Bind(wx.EVT_MENU, lambda e: self.ungroup(bid), item)
            else:
                item = menu.Append(wx.ID_ANY, "Remove from Diagram")
                self.Bind(wx.EVT_MENU, lambda e: self.remove_block(bid), item)

            self.PopupMenu(menu, event.GetPosition())
            menu.Destroy()

    def _on_motion(self, event):
        sx, sy = event.GetPosition()

        # Handle panning
        if self.panning:
            dx = sx - self.pan_start[0]
            dy = sy - self.pan_start[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.pan_start = (sx, sy)
            self.Refresh()
            return

        cx, cy = self.screen_to_canvas(sx, sy)

        if self.dragging and self.selected:
            b = self.blocks[self.selected]
            b.x, b.y = self._snap(cx - self.drag_offset[0], cy - self.drag_offset[1])
            b.x = max(0, b.x)
            b.y = max(0, b.y)
            self._update_group_bounds()
            self.Refresh()

        elif self.selecting:
            start_sx, start_sy = self.sel_start
            w, h = sx - start_sx, sy - start_sy
            if w < 0:
                start_sx, w = sx, -w
            if h < 0:
                start_sy, h = sy, -h
            self.sel_rect = (start_sx, start_sy, w, h)
            self.Refresh()

        else:
            old_hover = self.hover
            self.hover = self._block_at(cx, cy)
            if old_hover != self.hover:
                self.Refresh()

    def _on_mouse_wheel(self, event):
        """Zoom with mouse wheel."""
        rotation = event.GetWheelRotation()
        mouse_pos = event.GetPosition()

        if rotation > 0:
            self.zoom_in(center_on=mouse_pos)
        else:
            self.zoom_out(center_on=mouse_pos)

    def _on_middle_down(self, event):
        """Start panning with middle mouse button."""
        self.panning = True
        self.pan_start = event.GetPosition()
        self.SetCursor(wx.Cursor(wx.CURSOR_SIZING))
        self.CaptureMouse()

    def _on_middle_up(self, event):
        """Stop panning."""
        if self.panning:
            self.panning = False
            self.SetCursor(wx.Cursor(wx.CURSOR_DEFAULT))
            if self.HasCapture():
                self.ReleaseMouse()

    def _on_key(self, event):
        key = event.GetKeyCode()

        if key == wx.WXK_DELETE and self.selected:
            b = self.blocks.get(self.selected)
            if b and b.is_group:
                self.ungroup(self.selected)
            elif b:
                self.remove_block(self.selected)
        elif key in (ord('+'), ord('='), wx.WXK_NUMPAD_ADD):
            self.zoom_in()
        elif key in (ord('-'), wx.WXK_NUMPAD_SUBTRACT):
            self.zoom_out()
        elif key in (ord('f'), ord('F')):
            self.zoom_fit()
        elif key in (ord('0'), wx.WXK_NUMPAD0):
            self.zoom_reset()
        elif key == wx.WXK_LEFT:
            self.pan_offset[0] += 50
            self.Refresh()
        elif key == wx.WXK_RIGHT:
            self.pan_offset[0] -= 50
            self.Refresh()
        elif key == wx.WXK_UP:
            self.pan_offset[1] += 50
            self.Refresh()
        elif key == wx.WXK_DOWN:
            self.pan_offset[1] -= 50
            self.Refresh()
        else:
            event.Skip()

    def _show_group_menu(self, block_ids: List[str]):
        """Show menu to create group."""
        menu = wx.Menu()

        id_s = wx.NewId()
        id_p = wx.NewId()
        id_k = wx.NewId()

        menu.Append(id_s, "Group as SERIES (all must work)")
        menu.Append(id_p, "Group as PARALLEL (any can work)")
        menu.Append(id_k, f"Group as K-of-{len(block_ids)} (redundancy)...")

        self.Bind(wx.EVT_MENU, lambda e: self.create_group(block_ids, "series"), id=id_s)
        self.Bind(wx.EVT_MENU, lambda e: self.create_group(block_ids, "parallel"), id=id_p)

        def on_kn(e):
            dlg = wx.NumberEntryDialog(self, f"How many must work?", "K:",
                                       "K-of-N Redundancy", 2, 1, len(block_ids))
            if dlg.ShowModal() == wx.ID_OK:
                self.create_group(block_ids, "k_of_n", dlg.GetValue())
            dlg.Destroy()

        self.Bind(wx.EVT_MENU, on_kn, id=id_k)

        self.PopupMenu(menu)
        menu.Destroy()

    def _edit_group(self, group_id: str):
        """Edit group properties."""
        g = self.blocks.get(group_id)
        if not g or not g.is_group:
            return

        choices = ["SERIES (all must work)", "PARALLEL (any can work)",
                   f"K-of-{len(g.children)} (redundancy)"]

        dlg = wx.SingleChoiceDialog(self, "Select connection type:", "Group Type", choices)

        idx = {"series": 0, "parallel": 1, "k_of_n": 2}
        dlg.SetSelection(idx.get(g.connection_type, 0))

        if dlg.ShowModal() == wx.ID_OK:
            sel = dlg.GetSelection()
            if sel == 0:
                g.connection_type = "series"
                g.label = "SERIES"
            elif sel == 1:
                g.connection_type = "parallel"
                g.label = "PARALLEL"
            else:
                kdlg = wx.NumberEntryDialog(self, "How many must work?", "K:",
                                            "K-of-N", g.k_value, 1, len(g.children))
                if kdlg.ShowModal() == wx.ID_OK:
                    g.k_value = kdlg.GetValue()
                kdlg.Destroy()
                g.connection_type = "k_of_n"
                g.label = f"{g.k_value}-of-{len(g.children)}"

            self.Refresh()
            self._notify_change()

        dlg.Destroy()

    # === Data access ===

    def get_structure(self) -> Dict:
        """Get serializable structure."""
        return {
            "blocks": {
                bid: {
                    "name": b.name, "label": b.label,
                    "x": b.x, "y": b.y,
                    "is_group": b.is_group, "children": b.children,
                    "connection_type": b.connection_type,
                    "k_value": b.k_value,
                }
                for bid, b in self.blocks.items()
            },
            "root": self.root_id,
            "mission_hours": self.mission_hours,
        }

    def load_structure(self, data: Dict):
        """Load structure from dict."""
        self.blocks.clear()

        for bid, bd in data.get("blocks", {}).items():
            b = Block(
                id=bid, name=bd["name"], label=bd["label"],
                x=bd["x"], y=bd["y"],
                is_group=bd["is_group"], children=bd.get("children", []),
                connection_type=bd.get("connection_type", "series"),
                k_value=bd.get("k_value", 2)
            )
            self.blocks[bid] = b

        self.root_id = data.get("root")
        self.mission_hours = data.get("mission_hours", 5*365*24)
        self.Refresh()

    def update_block(self, block_id: str, r: float, lam: float):
        """Update block reliability values."""
        if block_id in self.blocks:
            self.blocks[block_id].reliability = r
            self.blocks[block_id].lambda_val = lam

    def clear(self):
        """Clear all blocks."""
        self.blocks.clear()
        self.root_id = None
        self.selected = None
        self.Refresh()
