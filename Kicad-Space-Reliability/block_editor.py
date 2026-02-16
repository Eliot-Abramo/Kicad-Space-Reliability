"""
Visual Block Diagram Editor
===========================
Drag-and-drop canvas for defining reliability topology with zoom/pan.

Author:  Eliot Abramo
"""

import wx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class Block:
    """A block in the reliability diagram."""
    id: str
    name: str
    label: str
    x: int = 0
    y: int = 0
    width: int = 180
    height: int = 70
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
    """Visual editor for reliability block diagrams."""

    GRID = 20
    PAD = 20

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
        self.SetMinSize((800, 400))

        self.blocks: Dict[str, Block] = {}
        self.root_id: Optional[str] = None
        self.selected: Optional[str] = None
        self.multi_selected: List[str] = []  # rubber-band multi selection
        self.hover: Optional[str] = None
        self.dragging = False
        self.drag_offset = (0, 0)
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.panning = False
        self.pan_start = (0, 0)
        self.mission_hours = 5 * 365 * 24

        # Rubber-band selection (Ctrl+left-drag on empty)
        self.rubber_band = False
        self.rubber_start = (0, 0)
        self.rubber_end = (0, 0)
        # Left-drag pan on empty space (no Ctrl)
        self.left_drag_pan = False

        self.on_selection_change = None
        self.on_structure_change = None
        self.on_block_activate = None
        self.on_zoom_change = None

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        self.Bind(wx.EVT_LEFT_DCLICK, self._on_dclick)
        self.Bind(wx.EVT_RIGHT_DOWN, self._on_right_click)
        self.Bind(wx.EVT_MIDDLE_DOWN, self._on_middle_down)
        self.Bind(wx.EVT_MIDDLE_UP, self._on_middle_up)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_mouse_wheel)
        self.Bind(wx.EVT_KEY_DOWN, self._on_key)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())
        self.SetFocus()

    def screen_to_canvas(self, sx, sy):
        return int((sx - self.pan_offset[0]) / self.zoom_level), int((sy - self.pan_offset[1]) / self.zoom_level)

    def canvas_to_screen(self, cx, cy):
        return int(cx * self.zoom_level + self.pan_offset[0]), int(cy * self.zoom_level + self.pan_offset[1])

    def set_zoom(self, zoom, center_on=None):
        old_zoom = self.zoom_level
        new_zoom = max(0.25, min(3.0, zoom))
        if abs(new_zoom - old_zoom) < 0.01: return
        if center_on:
            cx, cy = center_on
            canvas_x = (cx - self.pan_offset[0]) / old_zoom
            canvas_y = (cy - self.pan_offset[1]) / old_zoom
            self.pan_offset[0] = cx - canvas_x * new_zoom
            self.pan_offset[1] = cy - canvas_y * new_zoom
        self.zoom_level = new_zoom
        self.Refresh()
        if self.on_zoom_change: self.on_zoom_change(new_zoom)

    def zoom_fit(self):
        if not self.blocks: return
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for b in self.blocks.values():
            min_x, min_y = min(min_x, b.x), min(min_y, b.y)
            max_x, max_y = max(max_x, b.x + b.width), max(max_y, b.y + b.height)
        if min_x == float('inf'): return
        pad = 40
        content_w, content_h = max_x - min_x + pad*2, max_y - min_y + pad*2
        panel_w, panel_h = self.GetSize()
        if panel_w <= 0 or panel_h <= 0: return
        fit_zoom = min(panel_w / content_w, panel_h / content_h, 2.0)
        self.zoom_level = max(fit_zoom, 0.25)
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        self.pan_offset[0] = panel_w / 2 - center_x * self.zoom_level
        self.pan_offset[1] = panel_h / 2 - center_y * self.zoom_level
        self.Refresh()

    def add_block(self, block_id: str, name: str, label: str = None) -> Block:
        if label is None: label = name.rstrip('/').split('/')[-1] or name
        x, y = self._find_position()
        block = Block(id=block_id, name=name, label=label, x=x, y=y)
        self.blocks[block_id] = block
        if self.root_id is None:
            root = Block(id="__root__", name="System", label="System", is_group=True, connection_type="series")
            self.blocks["__root__"] = root
            self.root_id = "__root__"
        root = self.blocks.get(self.root_id)
        if root and block_id not in root.children:
            root.children.append(block_id)
        self._update_group_bounds()
        self.Refresh()
        return block

    def remove_block(self, block_id: str):
        if block_id not in self.blocks: return
        for b in self.blocks.values():
            if b.is_group and block_id in b.children:
                b.children.remove(block_id)
        del self.blocks[block_id]
        if self.selected == block_id: self.selected = None
        self._update_group_bounds()
        self.Refresh()
        self._notify_change()

    def create_group(self, block_ids: List[str], conn_type: str, k: int = 2) -> Optional[str]:
        if len(block_ids) < 2: return None
        gid = f"__grp_{sum(1 for b in self.blocks.values() if b.is_group)}__"
        min_x = min(self.blocks[bid].x for bid in block_ids)
        min_y = min(self.blocks[bid].y for bid in block_ids)
        max_x = max(self.blocks[bid].x + self.blocks[bid].width for bid in block_ids)
        max_y = max(self.blocks[bid].y + self.blocks[bid].height for bid in block_ids)
        label = {"series": "SERIES", "parallel": "PARALLEL", "k_of_n": f"{k}-of-{len(block_ids)}"}[conn_type]
        group = Block(id=gid, name=label, label=label, x=min_x - self.PAD, y=min_y - self.PAD,
                      width=max_x - min_x + 2*self.PAD, height=max_y - min_y + 2*self.PAD,
                      is_group=True, children=list(block_ids), connection_type=conn_type, k_value=k)
        self.blocks[gid] = group
        root = self.blocks.get(self.root_id)
        if root:
            for bid in block_ids:
                if bid in root.children: root.children.remove(bid)
            root.children.append(gid)
        self.Refresh()
        self._notify_change()
        return gid

    def ungroup(self, group_id: str):
        if group_id not in self.blocks or not self.blocks[group_id].is_group: return
        group = self.blocks[group_id]
        children = group.children.copy()
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
        x, y = self.PAD + self.GRID, self.PAD + self.GRID
        spacing_x, spacing_y = 220, 110
        max_x = 800
        while True:
            collision = any(not b.is_group and abs(b.x - x) < spacing_x and abs(b.y - y) < spacing_y for b in self.blocks.values())
            if not collision: return (x, y)
            x += spacing_x
            if x > max_x: x, y = self.PAD + self.GRID, y + spacing_y

    def _snap(self, x: int, y: int) -> Tuple[int, int]:
        return (round(x / self.GRID) * self.GRID, round(y / self.GRID) * self.GRID)

    def _block_at(self, x: int, y: int) -> Optional[str]:
        for bid, b in self.blocks.items():
            if not b.is_group and b.contains(x, y): return bid
        for bid, b in self.blocks.items():
            if b.is_group and b.contains(x, y): return bid
        return None

    def _update_group_bounds(self):
        for g in self.blocks.values():
            if g.is_group and g.children:
                min_x = min_y = float('inf')
                max_x = max_y = float('-inf')
                for cid in g.children:
                    c = self.blocks.get(cid)
                    if c:
                        min_x, min_y = min(min_x, c.x), min(min_y, c.y)
                        max_x, max_y = max(max_x, c.x + c.width), max(max_y, c.y + c.height)
                if min_x != float('inf'):
                    g.x, g.y = int(min_x - self.PAD), int(min_y - self.PAD)
                    g.width, g.height = int(max_x - min_x + 2*self.PAD), int(max_y - min_y + 2*self.PAD)

    def _notify_change(self):
        if self.on_structure_change: self.on_structure_change()

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        w, h = self.GetSize()
        gc.SetBrush(wx.Brush(self.BG))
        gc.DrawRectangle(0, 0, w, h)
        gc.Translate(self.pan_offset[0], self.pan_offset[1])
        gc.Scale(self.zoom_level, self.zoom_level)
        
        # Draw groups first
        for b in self.blocks.values():
            if b.is_group: self._draw_group(gc, b)
        # Then blocks
        for b in self.blocks.values():
            if not b.is_group: self._draw_block(gc, b)
        
        # Draw rubber-band selection rectangle
        if self.rubber_band:
            x1 = min(self.rubber_start[0], self.rubber_end[0])
            y1 = min(self.rubber_start[1], self.rubber_end[1])
            rw = abs(self.rubber_end[0] - self.rubber_start[0])
            rh = abs(self.rubber_end[1] - self.rubber_start[1])
            gc.SetBrush(wx.Brush(wx.Colour(100, 150, 255, 30)))
            gc.SetPen(wx.Pen(wx.Colour(50, 100, 200), 2, wx.PENSTYLE_SHORT_DASH))
            gc.DrawRectangle(x1, y1, rw, rh)

    def _draw_block(self, gc, b: Block):
        if b.id == self.selected:
            gc.SetBrush(wx.Brush(self.BLOCK_SEL))
            gc.SetPen(wx.Pen(wx.Colour(50, 100, 200), 3))
        elif b.id in self.multi_selected:
            gc.SetBrush(wx.Brush(wx.Colour(180, 200, 255)))
            gc.SetPen(wx.Pen(wx.Colour(80, 120, 200), 2, wx.PENSTYLE_SHORT_DASH))
        else:
            gc.SetBrush(wx.Brush(self.BLOCK_COLOR))
            gc.SetPen(wx.Pen(wx.Colour(100, 100, 100), 1))
        gc.DrawRoundedRectangle(b.x, b.y, b.width, b.height, 8)
        font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        gc.SetFont(font, wx.Colour(30, 30, 30))
        label = b.label[:18] + "..." if len(b.label) > 18 else b.label
        tw = gc.GetTextExtent(label)[0]
        gc.DrawText(label, b.x + (b.width - tw)/2, b.y + 10)
        font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font, wx.Colour(60, 60, 60))
        gc.DrawText(f"R = {b.reliability:.4f}", b.x + 10, b.y + 32)
        gc.DrawText(f"L = {b.lambda_val*1e9:.1f} FIT", b.x + 10, b.y + 48)

    def _draw_group(self, gc, g: Block):
        color = {"series": self.SERIES_COLOR, "parallel": self.PARALLEL_COLOR, "k_of_n": self.KN_COLOR}.get(g.connection_type, self.SERIES_COLOR)
        gc.SetBrush(wx.Brush(wx.Colour(color.Red(), color.Green(), color.Blue(), 60)))
        gc.SetPen(wx.Pen(wx.Colour(100, 100, 100), 2, wx.PENSTYLE_DOT))
        gc.DrawRoundedRectangle(g.x, g.y, g.width, g.height, 10)
        font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        gc.SetFont(font, wx.Colour(80, 80, 80))
        gc.DrawText(g.label, g.x + 6, g.y + 3)
        gc.DrawText(f"R={g.reliability:.4f}", g.x + g.width - 80, g.y + 3)

    def _on_left_down(self, event):
        sx, sy = event.GetPosition()
        cx, cy = self.screen_to_canvas(sx, sy)
        self.SetFocus()
        bid = self._block_at(cx, cy)
        if bid:
            if event.ShiftDown():
                # Shift+click: toggle multi-selection
                if bid in self.multi_selected:
                    self.multi_selected.remove(bid)
                else:
                    self.multi_selected.append(bid)
                self.selected = bid
            else:
                self.selected = bid
                if bid not in self.multi_selected:
                    self.multi_selected = [bid]
                self.dragging = True
                b = self.blocks[bid]
                self.drag_offset = (cx - b.x, cy - b.y)
            if self.on_selection_change: self.on_selection_change(self.selected)
        else:
            # Empty space: Ctrl+left = rubber-band, plain left = pan
            if not event.ShiftDown():
                self.selected = None
                self.multi_selected = []
            if event.ControlDown():
                self.rubber_band = True
                self.rubber_start = (cx, cy)
                self.rubber_end = (cx, cy)
            else:
                self.left_drag_pan = True
                self.pan_start = (sx, sy)
        self.Refresh()

    def _on_left_up(self, event):
        if self.rubber_band:
            # Finish rubber-band: select all non-group blocks in the rectangle
            self.rubber_band = False
            x1 = min(self.rubber_start[0], self.rubber_end[0])
            y1 = min(self.rubber_start[1], self.rubber_end[1])
            x2 = max(self.rubber_start[0], self.rubber_end[0])
            y2 = max(self.rubber_start[1], self.rubber_end[1])
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                for bid, b in self.blocks.items():
                    if not b.is_group:
                        bcx, bcy = b.center()
                        if x1 <= bcx <= x2 and y1 <= bcy <= y2:
                            if bid not in self.multi_selected:
                                self.multi_selected.append(bid)
                if self.multi_selected:
                    self.selected = self.multi_selected[-1]
                    if self.on_selection_change:
                        self.on_selection_change(self.selected)
            self.Refresh()
            return
        if self.left_drag_pan:
            self.left_drag_pan = False
            self.Refresh()
            return
        if self.dragging:
            self.dragging = False
            self._update_group_bounds()
            self._notify_change()
        self.panning = False
        self.Refresh()

    def _on_dclick(self, event):
        sx, sy = event.GetPosition()
        cx, cy = self.screen_to_canvas(sx, sy)
        bid = self._block_at(cx, cy)
        if bid:
            b = self.blocks[bid]
            if b.is_group: self._edit_group(bid)
            elif self.on_block_activate: self.on_block_activate(bid, b.name)

    def _on_right_click(self, event):
        sx, sy = event.GetPosition()
        cx, cy = self.screen_to_canvas(sx, sy)
        bid = self._block_at(cx, cy)
        
        menu = wx.Menu()
        
        # Multi-selection: allow grouping blocks and/or groups (nested: K-of-N in parallel, etc.)
        groupable = [b for b in self.multi_selected if b in self.blocks]
        if len(groupable) >= 2:
            item_s = menu.Append(wx.ID_ANY, f"Group {len(groupable)} blocks  Series")
            self.Bind(wx.EVT_MENU, lambda e, ids=groupable: self._group_selected(ids, "series"), item_s)
            item_p = menu.Append(wx.ID_ANY, f"Group {len(groupable)} blocks  Parallel")
            self.Bind(wx.EVT_MENU, lambda e, ids=groupable: self._group_selected(ids, "parallel"), item_p)
            item_k = menu.Append(wx.ID_ANY, f"Group {len(groupable)} blocks  K-of-N")
            self.Bind(wx.EVT_MENU, lambda e, ids=groupable: self._group_selected(ids, "k_of_n"), item_k)
            menu.AppendSeparator()
        
        if bid:
            self.selected = bid
            self.Refresh()
            b = self.blocks[bid]
            if b.is_group:
                item = menu.Append(wx.ID_ANY, "Edit Group...")
                self.Bind(wx.EVT_MENU, lambda e: self._edit_group(bid), item)
                menu.AppendSeparator()
                item = menu.Append(wx.ID_ANY, "Ungroup")
                self.Bind(wx.EVT_MENU, lambda e: self.ungroup(bid), item)
            else:
                item = menu.Append(wx.ID_ANY, "Remove")
                self.Bind(wx.EVT_MENU, lambda e: self.remove_block(bid), item)
        
        if menu.GetMenuItemCount() > 0:
            self.PopupMenu(menu, event.GetPosition())
        menu.Destroy()
    
    def _group_selected(self, block_ids: List[str], conn_type: str):
        """Group selected blocks with given connection type."""
        k = 2
        if conn_type == "k_of_n":
            dlg = wx.NumberEntryDialog(self, "K value:", "K:", "K-of-N", 2, 1, len(block_ids))
            if dlg.ShowModal() == wx.ID_OK:
                k = dlg.GetValue()
            else:
                dlg.Destroy()
                return
            dlg.Destroy()
        self.create_group(block_ids, conn_type, k)
        self.multi_selected = []

    def _on_motion(self, event):
        sx, sy = event.GetPosition()
        if self.rubber_band:
            cx, cy = self.screen_to_canvas(sx, sy)
            self.rubber_end = (cx, cy)
            self.Refresh()
            return
        if self.panning or self.left_drag_pan:
            self.pan_offset[0] += sx - self.pan_start[0]
            self.pan_offset[1] += sy - self.pan_start[1]
            self.pan_start = (sx, sy)
            self.Refresh()
            return
        cx, cy = self.screen_to_canvas(sx, sy)
        if self.dragging and self.selected:
            b = self.blocks[self.selected]
            b.x, b.y = self._snap(cx - self.drag_offset[0], cy - self.drag_offset[1])
            b.x, b.y = max(0, b.x), max(0, b.y)
            self._update_group_bounds()
            self.Refresh()
        else:
            old = self.hover
            self.hover = self._block_at(cx, cy)
            if old != self.hover: self.Refresh()

    def _on_middle_down(self, event):
        sx, sy = event.GetPosition()
        self.panning = True
        self.pan_start = (sx, sy)

    def _on_middle_up(self, event):
        self.panning = False

    def _on_mouse_wheel(self, event):
        rotation = event.GetWheelRotation()
        if rotation > 0: self.set_zoom(self.zoom_level + 0.1, event.GetPosition())
        else: self.set_zoom(self.zoom_level - 0.1, event.GetPosition())

    def _on_key(self, event):
        key = event.GetKeyCode()
        if key == wx.WXK_DELETE and self.selected:
            b = self.blocks.get(self.selected)
            if b and b.is_group: self.ungroup(self.selected)
            elif b: self.remove_block(self.selected)
        elif key in (ord('f'), ord('F')): self.zoom_fit()
        elif key == wx.WXK_ESCAPE:
            self.selected = None
            self.multi_selected = []
            self.Refresh()
        elif key == ord('G') or key == ord('g'):
            groupable = [b for b in self.multi_selected if b in self.blocks]
            if len(groupable) >= 2:
                self._group_selected(groupable, "series")
        else: event.Skip()

    def _edit_group(self, group_id: str):
        g = self.blocks.get(group_id)
        if not g or not g.is_group: return
        choices = ["SERIES", "PARALLEL", f"K-of-{len(g.children)}"]
        dlg = wx.SingleChoiceDialog(self, "Connection type:", "Group Type", choices)
        dlg.SetSelection({"series": 0, "parallel": 1, "k_of_n": 2}.get(g.connection_type, 0))
        if dlg.ShowModal() == wx.ID_OK:
            sel = dlg.GetSelection()
            if sel == 0: g.connection_type, g.label = "series", "SERIES"
            elif sel == 1: g.connection_type, g.label = "parallel", "PARALLEL"
            else:
                kdlg = wx.NumberEntryDialog(self, "K value:", "K:", "K-of-N", g.k_value, 1, len(g.children))
                if kdlg.ShowModal() == wx.ID_OK: g.k_value = kdlg.GetValue()
                kdlg.Destroy()
                g.connection_type, g.label = "k_of_n", f"{g.k_value}-of-{len(g.children)}"
            self.Refresh()
            self._notify_change()
        dlg.Destroy()

    def get_structure(self) -> Dict:
        return {
            "blocks": {bid: {"name": b.name, "label": b.label, "x": b.x, "y": b.y, "is_group": b.is_group,
                            "children": b.children, "connection_type": b.connection_type, "k_value": b.k_value}
                      for bid, b in self.blocks.items()},
            "root": self.root_id, "mission_hours": self.mission_hours,
        }

    def load_structure(self, data: Dict):
        self.blocks.clear()
        for bid, bd in data.get("blocks", {}).items():
            self.blocks[bid] = Block(id=bid, name=bd["name"], label=bd["label"], x=bd["x"], y=bd["y"],
                                     is_group=bd["is_group"], children=bd.get("children", []),
                                     connection_type=bd.get("connection_type", "series"), k_value=bd.get("k_value", 2))
        self.root_id = data.get("root")
        self.mission_hours = data.get("mission_hours", 43800)
        self.Refresh()

    def update_block(self, block_id: str, r: float, lam: float):
        if block_id in self.blocks:
            self.blocks[block_id].reliability = r
            self.blocks[block_id].lambda_val = lam

    def clear(self):
        self.blocks.clear()
        self.root_id = None
        self.selected = None
        self.Refresh()
