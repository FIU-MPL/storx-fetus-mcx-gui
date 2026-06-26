#!/usr/bin/env python3
"""
Fetus CT -> MCX GUI

Loads a MATLAB v7.3 CT label volume, visualizes orthogonal CT planes, remaps
the anatomical labels to 3, 5, or 7 optical layers, lets the user click a
source position near the maternal belly surface, and exports/runs an MCX/PMCX
configuration.

Coordinate convention used here:
    CT volume after loading:      vol_ct[x, y, z]
    CT z axis:                   neck -> legs
    CT y axis:                   anterior/posterior tissue depth
    MCX volume for photon run:    vol_mcx[x, z, y]
    MCX photon-depth axis:        mcx z = CT y
"""

import json
import os
import csv
import subprocess
import re
import itertools
import datetime
from pathlib import Path

import h5py
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm


# -------- Optical properties at 780 nm, copied from the user's examples --------
# MCX/PMCX property row order: [mua, mus, g, n], usually in mm^-1 for mua/mus
PROPS_780 = {
    3: [
        ("Background/Air",       0.0000,  0.0000, 1.0, 1.0000),
        ("Maternal Skin Bulk",   0.0406, 14.3840, 0.8, 1.4000),
        ("Maternal Muscle+Uterus", 0.0119, 8.6121, 0.9, 1.3700),
        ("Fetal Compartment",    0.0500, 10.0000, 0.9, 1.3700),
    ],
    5: [
        ("Background/Air",       0.0000,  0.0000, 1.0, 1.0000),
        ("Maternal Skin",        0.0406, 14.3840, 0.8, 1.4000),
        ("Maternal Adipose",     0.0022, 13.6470, 0.9, 1.3700),
        ("Maternal Muscle+Uterus", 0.0119, 8.6121, 0.9, 1.3700),
        ("Amniotic",             0.0020,  0.0100, 0.9, 1.3300),
        ("Fetal Compartment",    0.0500, 10.0000, 0.9, 1.3700),
    ],
    7: [
        ("Background/Air",       0.0000,  0.0000, 1.0, 1.0000),
        ("Maternal Skin",        0.0406, 14.3840, 0.8, 1.4000),
        ("Maternal Adipose",     0.0022, 13.6470, 0.9, 1.3700),
        ("Maternal Muscle",      0.0119,  8.6121, 0.9, 1.3700),
        ("Uterine Wall",         0.0120,  8.0000, 0.9, 1.3700),
        ("Amniotic",             0.0020,  0.0100, 0.9, 1.3300),
        ("Placenta",             0.1130,  8.8147, 0.9, 1.4000),
        ("Fetus",                0.0500, 10.0000, 0.9, 1.3700),
    ],
}


# Tissue IDs observed in Fetus_Model-1.mat
AIR = 99
REMAINDER = 10
SKIN = 11
ADIPOSE = 12
MUSCLE = 13
UTERUS = 88
PLACENTA = 90
AMNIOTIC = 92
FETAL_LABELS = set(range(110, 135))

# Maternal internal labels present in the file, grouped as maternal bulk/tissue.
MATERNAL_OTHER = {28, 32, 33, 46}  # kidney, liver, lung, cortical bone in this model


# Consistent display colors for layer IDs. These are used for the image layers,
# colored tissue boundaries, and the ID-column swatches in the optical-property table.
# Index 0 is air/background. Indices 1..7 are optical layer IDs.
LAYER_COLORS = {
    0: "#202020",  # background/air
    1: "#00d5ff",  # skin / skin bulk
    2: "#ffd400",  # adipose or muscle+uterus in 3-layer mode
    3: "#20e060",  # muscle / fetal compartment depending on mode
    4: "#d65cff",  # uterus or amniotic
    5: "#406bff",  # amniotic or fetal compartment
    6: "#ff3b30",  # placenta
    7: "#ffffff",  # fetus
}

# Layer color list used by matplotlib. Keep 0..7 order.
LAYER_COLOR_LIST = [LAYER_COLORS[i] for i in range(8)]
LAYER_CMAP = ListedColormap(LAYER_COLOR_LIST)
LAYER_NORM = BoundaryNorm(np.arange(-0.5, 8.5, 1.0), LAYER_CMAP.N)


def hex_to_qcolor(hex_color):
    return QtGui.QColor(hex_color)


def readable_text_color(hex_color):
    """Return black or white depending on background brightness."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return QtGui.QColor("#000000" if luminance > 150 else "#ffffff")


def load_mat_v73_volume(mat_path):
    """Load MATLAB v7.3 file and return vol_ct[x,y,z], voxel size in mm, and label names."""
    with h5py.File(mat_path, "r") as f:
        # MATLAB HDF5 arrays appear reversed in h5py.
        vol_h5 = np.asarray(f["volume"], dtype=np.uint8)      # shape read as [z,y,x]
        vol_ct = np.transpose(vol_h5, (2, 1, 0))              # now [x,y,z]

        voxel_size_cm = np.asarray(f["voxel_size_cm"]).ravel()
        voxel_size_mm = voxel_size_cm * 10.0

        ids = np.asarray(f["tissue_ids"]).ravel().astype(int)
        names = []
        for ref in f["tissue_names"][()].ravel():
            arr = np.asarray(f[ref]).squeeze()
            names.append("".join(chr(int(c)) for c in arr.flatten()))
        label_names = dict(zip(ids, names))

    return vol_ct, voxel_size_mm, label_names


def remap_to_layers(vol_ct, n_layers):
    """
    Return a remapped uint8 volume with labels:
        0 = air/background
        1..n_layers = optical tissues
    The rule follows the user's instruction:
        any remaining labeled tissue below the mother's skin is assigned to
        the closest available maternal group for the chosen simplification.
    """
    out = np.zeros_like(vol_ct, dtype=np.uint8)

    # Start with everything non-air as a maternal fallback, then overwrite.
    tissue = vol_ct != AIR

    if n_layers == 3:
        # 1 skin/bulk, 2 muscle+uterus, 3 fetal compartment
        out[tissue] = 1
        out[np.isin(vol_ct, [MUSCLE, UTERUS, ADIPOSE] + list(MATERNAL_OTHER))] = 2
        out[np.isin(vol_ct, [AMNIOTIC, PLACENTA] + list(FETAL_LABELS))] = 3
        out[vol_ct == SKIN] = 1
        out[vol_ct == REMAINDER] = 1

    elif n_layers == 5:
        # 1 skin, 2 adipose, 3 muscle+uterus, 4 amniotic, 5 fetal compartment
        out[tissue] = 3  # fallback below skin -> muscle/uterus group
        out[np.isin(vol_ct, [SKIN, REMAINDER])] = 1
        out[vol_ct == ADIPOSE] = 2
        out[np.isin(vol_ct, [MUSCLE, UTERUS] + list(MATERNAL_OTHER))] = 3
        out[vol_ct == AMNIOTIC] = 4
        out[np.isin(vol_ct, [PLACENTA] + list(FETAL_LABELS))] = 5

    elif n_layers == 7:
        # 1 skin, 2 adipose, 3 muscle, 4 uterus, 5 amniotic, 6 placenta, 7 fetus
        out[tissue] = 3  # fallback below skin -> muscle group
        out[np.isin(vol_ct, [SKIN, REMAINDER])] = 1
        out[vol_ct == ADIPOSE] = 2
        out[np.isin(vol_ct, [MUSCLE] + list(MATERNAL_OTHER))] = 3
        out[vol_ct == UTERUS] = 4
        out[vol_ct == AMNIOTIC] = 5
        out[vol_ct == PLACENTA] = 6
        out[np.isin(vol_ct, list(FETAL_LABELS))] = 7

    else:
        raise ValueError("n_layers must be 3, 5, or 7")

    return out


def ct_to_mcx_volume(vol_layers_ct):
    """Convert CT [x,y,z] to MCX [x,z,y], because MCX z-depth should equal CT y."""
    return np.transpose(vol_layers_ct, (0, 2, 1)).astype(np.uint8)


def ct_to_mcx_pos(x, y, z):
    """CT voxel coordinate [x,y,z] -> MCX coordinate [x,z,y]. MCX uses 1-based positions."""
    return [float(x + 1), float(z + 1), float(y + 1)]


class SliceCanvas(FigureCanvas):
    # Emitted for both click-to-place and drag-to-move source updates.
    sourceChanged = QtCore.pyqtSignal(str, float, float)

    def __init__(self, plane_name):
        self.fig = Figure(figsize=(4, 4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.plane_name = plane_name
        self.image = None
        self.cross_h = None
        self.cross_v = None
        self.source_marker = None
        self.source_xy = None
        self.dragging_source = False
        self.drag_pixel_radius = 14

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def set_source_xy(self, x, y):
        self.source_xy = (float(x), float(y))

    def _event_near_source(self, event):
        if self.source_xy is None or event.x is None or event.y is None:
            return False
        sx, sy = self.source_xy
        sx_pix, sy_pix = self.ax.transData.transform((sx, sy))
        dist = ((event.x - sx_pix) ** 2 + (event.y - sy_pix) ** 2) ** 0.5
        return dist <= self.drag_pixel_radius

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return

        # Clicking close to the current source starts a drag. Clicking anywhere
        # else places the source at the new clicked location.
        self.dragging_source = self._event_near_source(event)
        self.sourceChanged.emit(self.plane_name, float(event.xdata), float(event.ydata))

    def _on_motion(self, event):
        if not self.dragging_source:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.sourceChanged.emit(self.plane_name, float(event.xdata), float(event.ydata))

    def _on_release(self, event):
        self.dragging_source = False


def layer_props_for_wavelength(n_layers, wavelength_nm):
    """Return optical properties for wavelength. Currently 780 nm is calibrated; other wavelengths reuse 780 as placeholder."""
    # TODO: replace this with a measured/literature optical-property table keyed by wavelength.
    return PROPS_780[n_layers]


def default_volumes_folder():
    """Return the first existing default volumes folder.

    The intended project layout is:
        <project>/fetus_mcx_gui_v5.py
        <project>/volumes/Fetus_Model-1.mat
        <project>/volumes/Fetus_Model-2.mat
        ...
    When the script is run from a different working directory, this also checks
    the script directory so the GUI still finds the volumes folder.
    """
    candidates = [
        Path.cwd() / "volumes",
        Path(__file__).resolve().parent / "volumes",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return candidates[0]


def discover_patient_files(folder):
    folder = Path(folder).expanduser()
    if not folder.exists():
        return []

    def sort_key(p):
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else 10**9, p.name.lower())

    files = sorted(folder.glob("Fetus_Model-*.mat"), key=sort_key)
    # Fallback in case the downloaded dataset uses slightly different names.
    if not files:
        files = sorted(folder.glob("*.mat"), key=sort_key)
    return files


def make_range_values(start, stop, step, baseline=None):
    start = float(start); stop = float(stop); step = float(step)
    if step == 0:
        vals = [start]
    else:
        n = int(np.floor((stop - start) / step)) + 1 if step > 0 else int(np.floor((start - stop) / abs(step))) + 1
        vals = [start + i * step for i in range(max(n, 1))]
        vals = [v for v in vals if (v <= stop + 1e-9 if step > 0 else v >= stop - 1e-9)]
    if baseline is not None and not any(abs(v - baseline) < 1e-9 for v in vals):
        vals.append(float(baseline))
        vals = sorted(vals)
    return vals


def generate_transform_combos(params, mode="1D"):
    rx = make_range_values(*params["Rx"], baseline=0)
    ry = make_range_values(*params["Ry"], baseline=0)
    rz = make_range_values(*params["Rz"], baseline=0)
    tx = make_range_values(*params["Tx"], baseline=0)
    ty = make_range_values(*params["Ty"], baseline=0)
    tz = make_range_values(*params["Tz"], baseline=0)
    sc = make_range_values(*params["Scale"], baseline=100)
    if mode == "Grid":
        combos = list(itertools.product(rx, ry, rz, tx, ty, tz, sc))
    else:
        combos = [(0, 0, 0, 0, 0, 0, 100)]
        for v in rx:
            if abs(v) > 1e-9: combos.append((v, 0, 0, 0, 0, 0, 100))
        for v in ry:
            if abs(v) > 1e-9: combos.append((0, v, 0, 0, 0, 0, 100))
        for v in rz:
            if abs(v) > 1e-9: combos.append((0, 0, v, 0, 0, 0, 100))
        for v in tx:
            if abs(v) > 1e-9: combos.append((0, 0, 0, v, 0, 0, 100))
        for v in ty:
            if abs(v) > 1e-9: combos.append((0, 0, 0, 0, v, 0, 100))
        for v in tz:
            if abs(v) > 1e-9: combos.append((0, 0, 0, 0, 0, v, 100))
        for v in sc:
            if abs(v - 100) > 1e-9: combos.append((0, 0, 0, 0, 0, 0, v))
        combos = sorted(set(combos))
    return combos


def transform_id(combo):
    rx, ry, rz, tx, ty, tz, sc = combo
    return f"Rx{rx:g}_Ry{ry:g}_Rz{rz:g}_Tx{tx:g}_Ty{ty:g}_Tz{tz:g}_S{sc:g}".replace("-", "m").replace(".", "p")


def precompute_fetus_transform(vol_ct, voxel_size_mm, active_fetus=1):
    """Prepare fetal voxel coordinates for rigid/scale transforms in CT [x,y,z]."""
    if active_fetus == 1:
        fid_lo, fid_hi, brain_id = 110, 134, 113
    else:
        fid_lo, fid_hi, brain_id = 210, 234, 213

    fetal_mask = (vol_ct >= fid_lo) & (vol_ct <= fid_hi)
    base = vol_ct.copy()
    base[fetal_mask] = AMNIOTIC
    f_sub = np.column_stack(np.nonzero(fetal_mask))  # zero-based x,y,z
    if f_sub.size == 0:
        return base, f_sub, np.array([], dtype=vol_ct.dtype), np.zeros(3), np.eye(3)
    f_ids = vol_ct[tuple(f_sub.T)]

    coords_mm = (f_sub.astype(float) + 0.5) * voxel_size_mm[None, :]
    ctr_mm = coords_mm.mean(axis=0)

    brain_mask = (vol_ct == brain_id)
    if np.any(brain_mask):
        b_sub = np.column_stack(np.nonzero(brain_mask))
        brain_mm = ((b_sub.astype(float) + 0.5) * voxel_size_mm[None, :]).mean(axis=0)
    else:
        brain_mm = ctr_mm.copy()
        brain_mm[2] = coords_mm[:, 2].max()

    zp = brain_mm - ctr_mm
    if np.linalg.norm(zp) < 1e-9:
        zp = np.array([0.0, 0.0, 1.0])
    else:
        zp = zp / np.linalg.norm(zp)
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(zp, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    xp = np.cross(ref, zp); xp = xp / np.linalg.norm(xp)
    yp = np.cross(zp, xp)
    r_frame = np.column_stack([xp, yp, zp])
    return base, f_sub, f_ids, ctr_mm, r_frame


def apply_fetus_transform(base, f_sub, f_ids, voxel_size_mm, ctr_mm, r_frame, combo):
    """Apply Rx/Ry/Rz/Tx/Ty/Tz/Scale to fetus voxels only; translations are mm."""
    if f_sub.size == 0:
        return base.copy()
    rx, ry, rz, tx, ty, tz, scale_percent = combo
    coords_mm = (f_sub.astype(float) + 0.5) * voxel_size_mm[None, :]
    coords_c = coords_mm - ctr_mm[None, :]
    coords_h = coords_c @ r_frame
    scale = float(scale_percent) / 100.0
    ax, ay, az = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    R = Rz @ Ry @ Rx
    coords_h2 = (scale * coords_h) @ R.T + np.array([tx, ty, tz])[None, :]
    coords_new = coords_h2 @ r_frame.T + ctr_mm[None, :]
    new_sub = np.rint(coords_new / voxel_size_mm[None, :] - 0.5).astype(int)
    dims = np.array(base.shape)
    ok = np.all((new_sub >= 0) & (new_sub < dims[None, :]), axis=1)
    new_vol = base.copy()
    ns = new_sub[ok]
    if len(ns):
        new_vol[tuple(ns.T)] = f_ids[ok]
    return new_vol


def build_batch_cfg(vol_ct, voxel_size_mm, n_layers, wavelength_nm, source_ct, nphoton, det_sep_mm, det_radius_vox=10):
    layer_ct = remap_to_layers(vol_ct, n_layers)
    vol_mcx = ct_to_mcx_volume(layer_ct)
    props = [[mua, mus, g, n] for (_, mua, mus, g, n) in layer_props_for_wavelength(n_layers, wavelength_nm)]
    x, y, z = [int(v) for v in source_ct]
    ny_ct = vol_ct.shape[1]
    srcdir = [0.0, 0.0, 1.0] if y < ny_ct / 2 else [0.0, 0.0, -1.0]
    srcpos = ct_to_mcx_pos(x, y, z)
    dx_vox = int(round(float(det_sep_mm) / float(voxel_size_mm[0])))
    det_x = int(np.clip(x + dx_vox, 0, vol_ct.shape[0] - 1))
    detpos = [ct_to_mcx_pos(det_x, y, z) + [float(det_radius_vox)]]
    return {
        "nphoton": int(nphoton), "vol": vol_mcx,
        "tstart": 0.0, "tend": 5e-9, "tstep": 5e-9,
        "srcpos": srcpos, "srcdir": srcdir, "detpos": detpos,
        "prop": props, "unitinmm": float(voxel_size_mm[0]),
        "issrcfrom0": 0, "isreflect": 1, "isnormalized": 1,
        "autopilot": 1, "gpuid": 1,
    }


class FetusMCXGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fetus CT Optical Layer Mapper + MCX Launcher")
        self.resize(1600, 850)

        self.vol_ct = None
        self.layer_ct = None
        self.label_names = {}
        self.voxel_size_mm = np.array([2.0, 2.0, 1.0])  # fallback from inspected file

        self.x = 0
        self.y = 0
        self.z = 0
        self.source_ct = None
        self.patient_id = None
        self.current_mat_path = None
        self.n_layers = 7
        self.flux_ct = None
        self.flux_log_ct = None
        self.show_flux_overlay = False
        self.boundaries_only = False
        self.patient_files = []
        self.default_patient_folder = default_volumes_folder()
        self._loading_patient_combo = False

        # Batch MCX result browser state. We index result folders but load only
        # the currently selected result to avoid filling RAM with many flux volumes.
        self.batch_result_root = None
        self.batch_result_dirs = []
        self.batch_result_index = -1
        self._loading_result_combo = False

        self._build_ui()
        QtCore.QTimer.singleShot(0, self.load_default_patients_on_boot)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        main.addLayout(top)

        top.addWidget(QtWidgets.QLabel("Dataset:"))
        self.patient_combo = QtWidgets.QComboBox()
        self.patient_combo.setMinimumWidth(220)
        self.patient_combo.currentIndexChanged.connect(self.patient_combo_changed)
        top.addWidget(self.patient_combo)

        self.prev_patient_btn = QtWidgets.QPushButton("Previous")
        self.prev_patient_btn.clicked.connect(lambda: self.step_patient(-1))
        top.addWidget(self.prev_patient_btn)

        self.next_patient_btn = QtWidgets.QPushButton("Next")
        self.next_patient_btn.clicked.connect(lambda: self.step_patient(1))
        top.addWidget(self.next_patient_btn)

        self.reload_patient_list_btn = QtWidgets.QPushButton("Reload volumes")
        self.reload_patient_list_btn.clicked.connect(lambda: self.load_patient_folder(self.default_patient_folder, auto_load=True))
        top.addWidget(self.reload_patient_list_btn)

        self.load_btn = QtWidgets.QPushButton("Load .mat")
        self.load_btn.clicked.connect(self.load_mat)
        top.addWidget(self.load_btn)

        top.addWidget(QtWidgets.QLabel("Layer model:"))
        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.addItems(["3", "5", "7"])
        self.layer_combo.setCurrentText("7")
        self.layer_combo.currentTextChanged.connect(self.update_layer_model)
        top.addWidget(self.layer_combo)

        top.addWidget(QtWidgets.QLabel("Patient ID:"))
        self.patient_id_edit = QtWidgets.QLineEdit()
        self.patient_id_edit.setPlaceholderText("e.g., fetus-1")
        self.patient_id_edit.setMaximumWidth(160)
        self.patient_id_edit.editingFinished.connect(self.patient_id_changed)
        top.addWidget(self.patient_id_edit)

        self.save_source_btn = QtWidgets.QPushButton("Save source")
        self.save_source_btn.clicked.connect(self.save_source_location)
        top.addWidget(self.save_source_btn)

        self.load_source_btn = QtWidgets.QPushButton("Load source")
        self.load_source_btn.clicked.connect(self.load_saved_source_location)
        top.addWidget(self.load_source_btn)

        self.export_btn = QtWidgets.QPushButton("Export MCX files")
        self.export_btn.clicked.connect(self.export_mcx)
        top.addWidget(self.export_btn)

        self.run_btn = QtWidgets.QPushButton("Run PMCX")
        self.run_btn.clicked.connect(self.run_pmcx)
        top.addWidget(self.run_btn)

        self.batch_btn = QtWidgets.QPushButton("Batch source/sweep")
        self.batch_btn.clicked.connect(self.open_batch_dialog)
        top.addWidget(self.batch_btn)

        self.load_result_btn = QtWidgets.QPushButton("Load MCX result folder")
        self.load_result_btn.clicked.connect(self.load_mcx_result_folder)
        top.addWidget(self.load_result_btn)

        self.prev_result_btn = QtWidgets.QPushButton("Prev result")
        self.prev_result_btn.clicked.connect(lambda: self.step_batch_result(-1))
        self.prev_result_btn.setEnabled(False)
        top.addWidget(self.prev_result_btn)

        self.next_result_btn = QtWidgets.QPushButton("Next result")
        self.next_result_btn.clicked.connect(lambda: self.step_batch_result(1))
        self.next_result_btn.setEnabled(False)
        top.addWidget(self.next_result_btn)

        top.addWidget(QtWidgets.QLabel("Result:"))
        self.result_combo = QtWidgets.QComboBox()
        self.result_combo.setMinimumWidth(260)
        self.result_combo.setEnabled(False)
        self.result_combo.currentIndexChanged.connect(self.result_combo_changed)
        top.addWidget(self.result_combo)

        self.show_flux_check = QtWidgets.QCheckBox("Show photons/fluence")
        self.show_flux_check.setEnabled(False)
        self.show_flux_check.toggled.connect(self.toggle_flux_overlay)
        top.addWidget(self.show_flux_check)

        self.boundary_only_check = QtWidgets.QCheckBox("Boundaries only")
        self.boundary_only_check.toggled.connect(self.toggle_boundaries_only)
        top.addWidget(self.boundary_only_check)

        top.addSpacing(20)
        self.status = QtWidgets.QLabel("Load Fetus_Model-1.mat to begin.")
        top.addWidget(self.status, stretch=1)

        canv_row = QtWidgets.QHBoxLayout()
        main.addLayout(canv_row, stretch=1)

        self.axial = SliceCanvas("xy")
        self.coronal = SliceCanvas("xz")
        self.sagittal = SliceCanvas("yz")
        for c in [self.axial, self.coronal, self.sagittal]:
            c.sourceChanged.connect(self.handle_source_update)
            canv_row.addWidget(c, stretch=1)

        controls = QtWidgets.QGridLayout()
        main.addLayout(controls)

        self.x_slider = self._make_slider(controls, 0, "X")
        self.y_slider = self._make_slider(controls, 1, "Y / tissue depth")
        self.z_slider = self._make_slider(controls, 2, "Z / neck-to-legs")

        self.photon_count = QtWidgets.QSpinBox()
        self.photon_count.setRange(1000, 1000000000)
        self.photon_count.setValue(1000000)
        self.photon_count.setSingleStep(100000)
        controls.addWidget(QtWidgets.QLabel("Photons"), 3, 0)
        controls.addWidget(self.photon_count, 3, 1)

        self.det_sep = QtWidgets.QDoubleSpinBox()
        self.det_sep.setRange(1.0, 100.0)
        self.det_sep.setValue(30.0)
        self.det_sep.setSuffix(" mm")
        controls.addWidget(QtWidgets.QLabel("Detector offset from source along CT X"), 3, 2)
        controls.addWidget(self.det_sep, 3, 3)

        self.output_dir = QtWidgets.QLineEdit(str(Path.cwd() / "mcx_export"))
        controls.addWidget(QtWidgets.QLabel("Output folder"), 4, 0)
        controls.addWidget(self.output_dir, 4, 1, 1, 3)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Layer", "mua", "mus", "g/n"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        main.addWidget(self.table, stretch=0)
        self.refresh_prop_table()

    def load_default_patients_on_boot(self):
        """Scan ./volumes on boot and load the first patient automatically."""
        self.load_patient_folder(self.default_patient_folder, auto_load=True)

    def load_patient_folder(self, folder, auto_load=True):
        folder = Path(folder).expanduser()
        self.default_patient_folder = folder
        self.patient_files = discover_patient_files(folder)

        self._loading_patient_combo = True
        self.patient_combo.clear()
        for f in self.patient_files:
            self.patient_combo.addItem(f.name, str(f))
        self._loading_patient_combo = False

        if not self.patient_files:
            self.status.setText(
                f"No patient .mat files found in {folder}. Put the original CT volumes in a subfolder named 'volumes'."
            )
            return

        self.status.setText(f"Found {len(self.patient_files)} patient CT volumes in {folder}.")
        if auto_load:
            # Load the first patient. load_mat_path() will also restore a saved source if available.
            self.patient_combo.setCurrentIndex(0)
            self.load_mat_path(self.patient_files[0])

    def patient_combo_changed(self, index):
        if self._loading_patient_combo:
            return
        if index < 0 or index >= len(self.patient_files):
            return
        self.load_mat_path(self.patient_files[index])

    def step_patient(self, delta):
        n = self.patient_combo.count()
        if n == 0:
            return
        new_index = (self.patient_combo.currentIndex() + int(delta)) % n
        self.patient_combo.setCurrentIndex(new_index)

    def sync_patient_combo_to_path(self, mat_path):
        """Keep the combo box selection synchronized without recursively loading."""
        mat_path = str(Path(mat_path).resolve())
        for i in range(self.patient_combo.count()):
            data = self.patient_combo.itemData(i)
            if data and str(Path(data).resolve()) == mat_path:
                self._loading_patient_combo = True
                self.patient_combo.setCurrentIndex(i)
                self._loading_patient_combo = False
                return

    def toggle_flux_overlay(self, checked):
        self.show_flux_overlay = bool(checked)
        self.draw_all()

    def toggle_boundaries_only(self, checked):
        self.boundaries_only = bool(checked)
        self.draw_all()

    def load_flux_result(self, result_path):
        """Load PMCX flux and convert it from MCX [x,z,y] to CT [x,y,z]."""
        res = np.load(result_path, allow_pickle=True)
        if "flux" not in res.files:
            raise KeyError(f"No 'flux' field found in {result_path}. Found: {res.files}")
        flux = np.squeeze(res["flux"])
        if flux.ndim == 4:
            flux = np.sum(flux, axis=-1)
        if flux.ndim != 3:
            raise ValueError(f"Expected 3D flux after squeeze/sum, got shape {flux.shape}")
        # PMCX/MCX volume orientation is [CT X, CT Z, CT Y]. Convert to CT [X,Y,Z].
        self.flux_ct = np.transpose(flux, (0, 2, 1))
        self.flux_log_ct = np.log10(self.flux_ct + 1e-30)
        self.show_flux_check.setEnabled(True)
        self.show_flux_check.setChecked(True)
        self.show_flux_overlay = True

    def _make_slider(self, layout, row, label):
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setRange(0, 1)
        s.valueChanged.connect(self.slider_changed)
        layout.addWidget(s, row, 1, 1, 3)
        return s

    def refresh_prop_table(self):
        props = PROPS_780[self.n_layers]
        self.table.setRowCount(len(props))
        for i, (name, mua, mus, g, n) in enumerate(props):
            vals = [str(i), name, f"{mua:.4f}", f"{mus:.4f}", f"{g:.2f} / {n:.3f}"]
            for j, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(val)
                if j == 0:
                    color = LAYER_COLORS.get(i, "#808080")
                    item.setBackground(hex_to_qcolor(color))
                    item.setForeground(readable_text_color(color))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(i, j, item)

    def infer_patient_id(self, mat_path):
        stem = Path(mat_path).stem.lower()
        stem = stem.replace("_", "-").replace(" ", "-")
        stem = re.sub(r"-+", "-", stem)
        # Common file names such as Fetus_Model-1.mat become fetus-1.
        stem = re.sub(r"fetus-?model-?(\d+)", r"fetus-\1", stem)
        return stem

    def source_store_path(self):
        outdir = Path(self.output_dir.text()).expanduser()
        return outdir / "source_locations.json"

    def patient_id_changed(self):
        text = self.patient_id_edit.text().strip()
        if text:
            self.patient_id = text

    def read_source_store(self):
        path = self.source_store_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def write_source_store(self, store):
        path = self.source_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(store, f, indent=2)

    def save_source_location(self):
        if self.source_ct is None:
            self.status.setText("No source selected yet. Click or drag the source on one of the planes first.")
            return
        patient_id = (self.patient_id_edit.text().strip() or self.patient_id or "unknown")
        self.patient_id = patient_id
        x, y, z = [int(v) for v in self.source_ct]
        store = self.read_source_store()
        store[patient_id] = {
            "source_ct": [x, y, z],
            "source_mcx_0_based": [x, z, y],
            "source_mcx_1_based": ct_to_mcx_pos(x, y, z),
            "mat_file": str(self.current_mat_path) if self.current_mat_path else None,
            "ct_shape": list(self.vol_ct.shape) if self.vol_ct is not None else None,
            "voxel_size_mm": self.voxel_size_mm.tolist(),
        }
        self.write_source_store(store)
        self.status.setText(
            f"Saved source for {patient_id}: CT [x,y,z]=[{x},{y},{z}], "
            f"MCX 0-based [x,y,z]=[{x},{z},{y}]."
        )

    def load_saved_source_location(self):
        if self.vol_ct is None:
            self.status.setText("Load a patient .mat file before loading a saved source.")
            return False
        patient_id = (self.patient_id_edit.text().strip() or self.patient_id or "").strip()
        if not patient_id:
            self.status.setText("Enter a Patient ID first, for example fetus-1.")
            return False
        store = self.read_source_store()
        if patient_id not in store:
            self.status.setText(f"No saved source found for {patient_id} in {self.source_store_path()}.")
            return False

        src = store[patient_id].get("source_ct")
        if not src or len(src) != 3:
            self.status.setText(f"Saved source for {patient_id} is invalid.")
            return False

        nx, ny, nz = self.vol_ct.shape
        x = int(np.clip(src[0], 0, nx - 1))
        y = int(np.clip(src[1], 0, ny - 1))
        z = int(np.clip(src[2], 0, nz - 1))
        self.set_source_ct(x, y, z, save=False, message_prefix=f"Loaded saved source for {patient_id}")
        return True

    def load_mat_path(self, fn):
        fn = str(fn)
        self.current_mat_path = Path(fn)
        self.patient_id = self.infer_patient_id(fn)
        self.patient_id_edit.setText(self.patient_id)
        self.sync_patient_combo_to_path(fn)

        self.vol_ct, self.voxel_size_mm, self.label_names = load_mat_v73_volume(fn)
        nx, ny, nz = self.vol_ct.shape

        self.x, self.y, self.z = nx // 2, ny // 2, nz // 2
        self.source_ct = None
        self.flux_ct = None
        self.flux_log_ct = None
        self.show_flux_overlay = False
        self.show_flux_check.setChecked(False)
        self.show_flux_check.setEnabled(False)
        self.x_slider.setRange(0, nx - 1)
        self.y_slider.setRange(0, ny - 1)
        self.z_slider.setRange(0, nz - 1)
        self.x_slider.setValue(self.x)
        self.y_slider.setValue(self.y)
        self.z_slider.setValue(self.z)

        self.update_layer_model(self.layer_combo.currentText())
        loaded_source = self.load_saved_source_location()
        if not loaded_source:
            self.status.setText(
                f"Loaded {Path(fn).name}: CT shape [X,Y,Z]={self.vol_ct.shape}, "
                f"voxel size={self.voxel_size_mm} mm. Click near the belly surface to set the source."
            )

    def load_mat(self):
        start_dir = str(self.default_patient_folder if self.default_patient_folder.exists() else Path.cwd())
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open MATLAB v7.3 CT file", start_dir, "MAT files (*.mat)"
        )
        if not fn:
            return
        self.load_mat_path(fn)

    def update_layer_model(self, text):
        self.n_layers = int(text)
        self.refresh_prop_table()
        if self.vol_ct is not None:
            self.layer_ct = remap_to_layers(self.vol_ct, self.n_layers)
            self.draw_all()

    def slider_changed(self):
        if self.vol_ct is None:
            return
        self.x = self.x_slider.value()
        self.y = self.y_slider.value()
        self.z = self.z_slider.value()
        self.draw_all()

    def set_source_ct(self, x, y, z, save=True, message_prefix="Source selected"):
        if self.vol_ct is None:
            return

        nx, ny, nz = self.vol_ct.shape
        self.x = int(np.clip(round(x), 0, nx - 1))
        self.y = int(np.clip(round(y), 0, ny - 1))
        self.z = int(np.clip(round(z), 0, nz - 1))
        self.source_ct = (self.x, self.y, self.z)

        self.x_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.z_slider.blockSignals(True)
        self.x_slider.setValue(self.x)
        self.y_slider.setValue(self.y)
        self.z_slider.setValue(self.z)
        self.x_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.z_slider.blockSignals(False)

        self.status.setText(
            f"{message_prefix}: CT [x,y,z]=[{self.x},{self.y},{self.z}], "
            f"MCX 0-based [x,y,z]=[{self.x},{self.z},{self.y}], "
            f"MCX 1-based srcpos={ct_to_mcx_pos(self.x, self.y, self.z)}."
        )
        self.draw_all()
        if save:
            self.save_source_location()

    def handle_source_update(self, plane, a, b):
        if self.vol_ct is None:
            return

        # Canvas events are in millimeters because the displayed axes use mm.
        vx, vy, vz = self.voxel_size_mm
        x, y, z = self.x, self.y, self.z

        if plane == "xy":
            x = a / vx
            y = b / vy
        elif plane == "xz":
            x = a / vx
            z = b / vz
        elif plane == "yz":
            y = a / vy
            z = b / vz

        self.set_source_ct(x, y, z, save=True, message_prefix="Source updated")

    def _axis_extent_mm(self, plane):
        """Return image extent and axis labels for a CT plane in mm."""
        nx, ny, nz = self.layer_ct.shape
        vx, vy, vz = self.voxel_size_mm
        if plane == "xy":
            return [0, nx * vx, 0, ny * vy], "CT X (mm)", "CT Y (mm)", self.x * vx, self.y * vy
        if plane == "xz":
            return [0, nx * vx, 0, nz * vz], "CT X (mm)", "CT Z (mm)", self.x * vx, self.z * vz
        if plane == "yz":
            return [0, ny * vy, 0, nz * vz], "CT Y (mm)", "CT Z (mm)", self.y * vy, self.z * vz
        raise ValueError(plane)

    def draw_all(self):
        if self.layer_ct is None:
            return

        flux_xy = flux_xz = flux_yz = None
        if self.show_flux_overlay and self.flux_log_ct is not None:
            # flux_log_ct is [CT X, CT Y, CT Z]
            flux_xy = self.flux_log_ct[:, :, self.z].T
            flux_xz = self.flux_log_ct[:, self.y, :].T
            flux_yz = self.flux_log_ct[self.x, :, :].T

        # Display remapped optical layers in CT coordinates.
        self._draw_canvas(
            self.axial,
            self.layer_ct[:, :, self.z].T,
            "CT XY plane",
            "xy",
            flux_img=flux_xy,
        )
        self._draw_canvas(
            self.coronal,
            self.layer_ct[:, self.y, :].T,
            "CT XZ plane",
            "xz",
            flux_img=flux_xz,
        )
        self._draw_canvas(
            self.sagittal,
            self.layer_ct[self.x, :, :].T,
            "CT YZ plane",
            "yz",
            flux_img=flux_yz,
        )

    def _draw_layer_boundaries(self, ax, label_img, extent):
        """Draw one colored contour per layer ID using mm coordinates."""
        x0, x1, y0, y1 = extent
        height, width = label_img.shape
        xs = np.linspace(x0, x1, width)
        ys = np.linspace(y0, y1, height)
        X, Y = np.meshgrid(xs, ys)
        for layer_id in sorted(np.unique(label_img.astype(int))):
            if layer_id == 0:
                continue
            color = LAYER_COLORS.get(int(layer_id), "#ffffff")
            mask = (label_img == layer_id).astype(float)
            # If this layer is absent or touches no transition, contour may fail; skip safely.
            if mask.max() <= 0:
                continue
            try:
                ax.contour(X, Y, mask, levels=[0.5], colors=[color], linewidths=1.2)
            except Exception:
                pass

    def _draw_canvas(self, canvas, img, title, plane, flux_img=None):
        ax = canvas.ax
        ax.clear()

        extent, xlabel, ylabel, vx_mm, vy_mm = self._axis_extent_mm(plane)

        if self.boundaries_only:
            # Plain background; anatomy is represented by ID-colored boundaries only.
            ax.imshow(
                np.zeros_like(img), origin="lower", interpolation="nearest",
                extent=extent, cmap="gray", vmin=0, vmax=1, alpha=0.08
            )
        else:
            ax.imshow(
                img, origin="lower", interpolation="nearest", extent=extent,
                cmap=LAYER_CMAP, norm=LAYER_NORM, alpha=0.85
            )

        if flux_img is not None:
            finite = np.isfinite(flux_img)
            if np.any(finite):
                positive = flux_img[finite]
                # Robust contrast: ignore extreme low zeros and very high outliers.
                vmin = np.percentile(positive, 5)
                vmax = np.percentile(positive, 99.5)
                if vmax <= vmin:
                    vmin, vmax = np.min(positive), np.max(positive)
                ax.imshow(
                    flux_img, origin="lower", interpolation="nearest", extent=extent,
                    cmap="inferno", alpha=0.68, vmin=vmin, vmax=vmax
                )

        # Always show colored boundaries so the table ID colors match what is displayed.
        self._draw_layer_boundaries(ax, img, extent)

        ax.axvline(vx_mm, linewidth=1, color="white", alpha=0.75)
        ax.axhline(vy_mm, linewidth=1, color="white", alpha=0.75)

        if self.source_ct is None:
            ax.plot(vx_mm, vy_mm, marker="+", markersize=10, color="white")
            canvas.set_source_xy(vx_mm, vy_mm)
        else:
            ax.plot(
                vx_mm, vy_mm, marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor="cyan", markeredgewidth=2.5
            )
            ax.plot(vx_mm, vy_mm, marker="+", markersize=12, color="cyan")
            canvas.set_source_xy(vx_mm, vy_mm)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        canvas.draw_idle()

    def _source_and_detector(self):
        if self.source_ct is None:
            raise RuntimeError("Click a source location near the maternal belly surface first.")

        x, y, z = self.source_ct
        nx, ny, nz = self.vol_ct.shape

        # Decide whether source is on low-y or high-y body surface.
        if y < ny / 2:
            srcdir = [0.0, 0.0, 1.0]    # +MCX z = +CT y
        else:
            srcdir = [0.0, 0.0, -1.0]   # -MCX z = -CT y

        srcpos = ct_to_mcx_pos(x, y, z)

        # Simple detector offset along CT x, converted to MCX x.
        dx_vox = int(round(self.det_sep.value() / float(self.voxel_size_mm[0])))
        det_x = int(np.clip(x + dx_vox, 0, nx - 1))
        detpos = ct_to_mcx_pos(det_x, y, z) + [3.0]  # [x,y,z,radius_vox]

        return srcpos, srcdir, [detpos]

    def build_cfg(self):
        vol_mcx = ct_to_mcx_volume(self.layer_ct)
        props = [[mua, mus, g, n] for (_, mua, mus, g, n) in PROPS_780[self.n_layers]]
        srcpos, srcdir, detpos = self._source_and_detector()

        cfg = {
            "nphoton": int(self.photon_count.value()),
            "vol": vol_mcx,
            "tstart": 0.0,
            "tend": 5e-9,
            "tstep": 5e-9,
            "srcpos": srcpos,
            "srcdir": srcdir,
            "detpos": detpos,
            "prop": props,
            "unitinmm": float(self.voxel_size_mm[0]),
            "issrcfrom0": 0,
            "isreflect": 1,
            "isnormalized": 1,
            "autopilot": 1,
            "gpuid": 1,
        }
        return cfg

    def export_mcx(self):
        if self.layer_ct is None:
            QtWidgets.QMessageBox.warning(self, "No volume", "Load a .mat file first.")
            return
        try:
            cfg = self.build_cfg()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Source needed", str(e))
            return

        outdir = Path(self.output_dir.text()).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)

        vol_mcx = cfg["vol"]
        bin_path = outdir / f"fetus_{self.n_layers}layer_mcx_volume.bin"
        vol_mcx.tofile(bin_path)

        # JSON export for command-line MCX. For PMCX, the GUI directly passes the NumPy volume.
        json_cfg = {k: v for k, v in cfg.items() if k != "vol"}
        json_cfg["Domain"] = {
            "Dim": list(vol_mcx.shape),
            "Media": [
                {"mua": p[0], "mus": p[1], "g": p[2], "n": p[3]}
                for p in cfg["prop"]
            ],
            "VolumeFile": str(bin_path),
        }
        json_cfg["Session"] = {"Photons": cfg["nphoton"], "ID": f"fetus_{self.n_layers}layer"}
        json_cfg["Forward"] = {"T0": cfg["tstart"], "T1": cfg["tend"], "Dt": cfg["tstep"]}
        json_cfg["Optode"] = {
            "Source": {"Pos": cfg["srcpos"], "Dir": cfg["srcdir"], "Type": "pencil"},
            "Detector": [{"Pos": d[:3], "R": d[3]} for d in cfg["detpos"]],
        }

        with open(outdir / f"fetus_{self.n_layers}layer_mcx.json", "w") as f:
            json.dump(json_cfg, f, indent=2)

        with open(outdir / f"optical_properties_{self.n_layers}layer_780nm.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mcx_id", "layer", "mua_mm^-1", "mus_mm^-1", "g", "n"])
            for i, row in enumerate(PROPS_780[self.n_layers]):
                writer.writerow([i, *row])

        np.save(outdir / f"fetus_{self.n_layers}layer_mcx_volume.npy", vol_mcx)

        # Keep a patient-specific copy of the selected source with each MCX export.
        if self.source_ct is not None:
            x, y, z = [int(v) for v in self.source_ct]
            with open(outdir / f"source_location_{self.patient_id or 'patient'}.json", "w") as f:
                json.dump({
                    "patient_id": self.patient_id or self.patient_id_edit.text().strip(),
                    "source_ct": [x, y, z],
                    "source_mcx_0_based": [x, z, y],
                    "source_mcx_1_based": ct_to_mcx_pos(x, y, z),
                    "coordinate_note": "CT [x,y,z] -> MCX [x,z,y]; MCX z is CT y depth",
                }, f, indent=2)

        self.status.setText(f"Exported MCX files to {outdir}")


    def load_mcx_result_folder(self):
        """Open one MCX result folder or a root containing many batch result folders."""
        start_dir = Path(self.default_patient_folder).expanduser() / "batch_mcx_results"
        if not start_dir.exists():
            start_dir = Path(self.output_dir.text()).expanduser()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select MCX result folder or batch_mcx_results root",
            str(start_dir)
        )
        if not folder:
            return

        folder = Path(folder).expanduser()
        direct_result = folder / "pmcx_result.npz"

        if direct_result.exists():
            self.set_batch_result_collection([folder], root=folder.parent)
            self.load_batch_result_by_index(0)
            return

        result_files = sorted(folder.rglob("pmcx_result.npz"))
        if not result_files:
            QtWidgets.QMessageBox.warning(
                self,
                "No MCX results found",
                f"No pmcx_result.npz files were found in:\n{folder}"
            )
            return

        result_dirs = [r.parent for r in result_files]
        self.set_batch_result_collection(result_dirs, root=folder)
        self.load_batch_result_by_index(0)

    def set_batch_result_collection(self, result_dirs, root=None):
        """Index a group of result directories for Next/Previous browsing."""
        self.batch_result_root = Path(root).expanduser() if root is not None else None
        self.batch_result_dirs = [Path(d).expanduser() for d in result_dirs]
        self.batch_result_index = -1

        self._loading_result_combo = True
        self.result_combo.clear()
        for i, d in enumerate(self.batch_result_dirs):
            self.result_combo.addItem(self.describe_batch_result_dir(d, i), str(d))
        self._loading_result_combo = False

        has_results = len(self.batch_result_dirs) > 0
        self.result_combo.setEnabled(has_results)
        self.prev_result_btn.setEnabled(len(self.batch_result_dirs) > 1)
        self.next_result_btn.setEnabled(len(self.batch_result_dirs) > 1)

        if has_results:
            self.status.setText(f"Indexed {len(self.batch_result_dirs)} MCX result folders. Use Prev result / Next result to browse.")

    def describe_batch_result_dir(self, result_dir, index=None):
        """Short readable label for the result combo box."""
        result_dir = Path(result_dir)
        meta = {}
        meta_path = result_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        patient = meta.get("patient_id", result_dir.parent.name)
        wav = meta.get("wavelength_nm", "?")
        layers = meta.get("layers", "?")
        transform = meta.get("transform_combo", {})

        if isinstance(transform, dict) and transform:
            tparts = []
            for k in ["Rx", "Ry", "Rz", "Tx", "Ty", "Tz", "Scale"]:
                if k in transform:
                    tparts.append(f"{k}={transform[k]}")
            transform_text = ", ".join(tparts) if tparts else result_dir.name
        else:
            transform_text = result_dir.name

        prefix = f"{index + 1}. " if index is not None else ""
        return f"{prefix}{patient} | {wav} nm | {layers} layers | {transform_text}"

    def result_combo_changed(self, index):
        if self._loading_result_combo:
            return
        if index < 0 or index >= len(self.batch_result_dirs):
            return
        if index == self.batch_result_index:
            return
        self.load_batch_result_by_index(index)

    def step_batch_result(self, delta):
        if not self.batch_result_dirs:
            QtWidgets.QMessageBox.information(self, "No indexed results", "Click 'Load MCX result folder' and select a batch_mcx_results root first.")
            return
        n = len(self.batch_result_dirs)
        if self.batch_result_index < 0:
            new_index = 0
        else:
            new_index = (self.batch_result_index + int(delta)) % n
        self.load_batch_result_by_index(new_index)

    def load_batch_result_by_index(self, index):
        if index < 0 or index >= len(self.batch_result_dirs):
            return
        self.batch_result_index = int(index)
        self.result_combo.blockSignals(True)
        self.result_combo.setCurrentIndex(index)
        self.result_combo.blockSignals(False)
        self.load_batch_result_directory(self.batch_result_dirs[index])

    def load_batch_result_directory(self, result_dir):
        """Load one batch result directory containing pmcx_result.npz and geometry files."""
        result_dir = Path(result_dir).expanduser()
        result_path = result_dir / "pmcx_result.npz"
        meta_path = result_dir / "metadata.json"
        transformed_path = result_dir / "transformed_volume_ct.npy"
        layer_mcx_path = result_dir / "mcx_volume_layers.npy"

        if not result_path.exists():
            QtWidgets.QMessageBox.warning(self, "Missing result", f"Could not find:\n{result_path}")
            return

        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Metadata warning", f"Could not read metadata.json:\n{e}")

        # Patient/layer/wavelength metadata for the UI.
        self.current_mat_path = result_dir
        self.patient_id = str(meta.get("patient_id", result_dir.parent.name))
        self.patient_id_edit.setText(self.patient_id)
        n_layers = int(meta.get("layers", self.n_layers))
        self.n_layers = n_layers
        self.layer_combo.blockSignals(True)
        self.layer_combo.setCurrentText(str(n_layers))
        self.layer_combo.blockSignals(False)
        self.refresh_prop_table()

        if "voxel_size_mm" in meta:
            try:
                self.voxel_size_mm = np.asarray(meta["voxel_size_mm"], dtype=float)
            except Exception:
                pass

        # Prefer the original transformed CT labels if available, so changing 3/5/7 layers still works.
        if transformed_path.exists():
            self.vol_ct = np.load(transformed_path)
            self.layer_ct = remap_to_layers(self.vol_ct, self.n_layers)
        elif layer_mcx_path.exists():
            # Stored batch MCX geometry is [CT X, CT Z, CT Y]; convert to CT display [X,Y,Z].
            layer_mcx = np.load(layer_mcx_path)
            self.layer_ct = np.transpose(layer_mcx, (0, 2, 1)).astype(np.uint8)
            self.vol_ct = self.layer_ct.copy()
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing geometry",
                "This folder has pmcx_result.npz but no transformed_volume_ct.npy or mcx_volume_layers.npy.\n"
                "The photon flux can be loaded, but there is no geometry to overlay it on."
            )
            # Load flux once to get shape, then use blank geometry.
            tmp = np.load(result_path, allow_pickle=True)
            flux = np.squeeze(tmp["flux"])
            if flux.ndim == 4:
                flux = np.sum(flux, axis=-1)
            self.layer_ct = np.zeros(np.transpose(flux, (0, 2, 1)).shape, dtype=np.uint8)
            self.vol_ct = self.layer_ct.copy()

        nx, ny, nz = self.layer_ct.shape
        self.x_slider.setRange(0, nx - 1)
        self.y_slider.setRange(0, ny - 1)
        self.z_slider.setRange(0, nz - 1)

        src = meta.get("source_ct")
        if src and len(src) == 3:
            self.x = int(np.clip(src[0], 0, nx - 1))
            self.y = int(np.clip(src[1], 0, ny - 1))
            self.z = int(np.clip(src[2], 0, nz - 1))
            self.source_ct = (self.x, self.y, self.z)
        else:
            self.x, self.y, self.z = nx // 2, ny // 2, nz // 2
            self.source_ct = None

        self.x_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.z_slider.blockSignals(True)
        self.x_slider.setValue(self.x)
        self.y_slider.setValue(self.y)
        self.z_slider.setValue(self.z)
        self.x_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.z_slider.blockSignals(False)

        try:
            self.load_flux_result(result_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Could not load photon flux", str(e))
            return

        if self.flux_log_ct is not None and self.flux_log_ct.shape != self.layer_ct.shape:
            QtWidgets.QMessageBox.warning(
                self,
                "Shape mismatch",
                f"Flux shape {self.flux_log_ct.shape} does not match geometry shape {self.layer_ct.shape}.\n"
                "The overlay may not align."
            )

        self.show_flux_check.setEnabled(True)
        self.show_flux_check.setChecked(True)
        self.show_flux_overlay = True
        self.draw_all()

        wav = meta.get("wavelength_nm", "unknown")
        transform = meta.get("transform_combo", {})
        transform_text = ", ".join(f"{k}={v}" for k, v in transform.items()) if transform else result_dir.name
        idx_text = ""
        if self.batch_result_dirs and 0 <= self.batch_result_index < len(self.batch_result_dirs):
            idx_text = f" [{self.batch_result_index + 1}/{len(self.batch_result_dirs)}]"
        self.status.setText(
            f"Loaded MCX result{idx_text}: {self.patient_id}, {wav} nm, {self.n_layers} layers, {transform_text} | {result_dir}"
        )

    def open_batch_dialog(self):
        dlg = BatchSweepDialog(self)
        dlg.exec_()

    def run_pmcx(self):
        if self.layer_ct is None:
            QtWidgets.QMessageBox.warning(self, "No volume", "Load a .mat file first.")
            return
        try:
            import pmcx
            cfg = self.build_cfg()
            self.status.setText("Running PMCX... this can take a while.")
            QtWidgets.QApplication.processEvents()
            res = pmcx.mcxlab(cfg)
            outdir = Path(self.output_dir.text()).expanduser()
            outdir.mkdir(parents=True, exist_ok=True)
            result_path = outdir / f"pmcx_result_{self.n_layers}layer.npz"
            np.savez_compressed(result_path, **res)
            self.load_flux_result(result_path)
            self.draw_all()
            self.status.setText(
                f"PMCX complete. Result saved in {outdir}. "
                "Use 'Show photons/fluence' and 'Boundaries only' to change the display."
            )
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self,
                "PMCX not installed",
                "Install PMCX in the same Python environment, e.g.:\n\npip install pmcx",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "PMCX error", str(e))


class BatchSweepDialog(QtWidgets.QDialog):
    """Batch source-selection and MCX sweep dialog for 46-patient fetal dataset expansion."""
    def __init__(self, parent_gui):
        super().__init__(parent_gui)
        self.parent_gui = parent_gui
        self.setWindowTitle("Batch patient source selection + transform/MCX sweep")
        self.resize(950, 620)
        self.files = []
        self._build_ui()
        if parent_gui.current_mat_path is not None:
            self.folder_edit.setText(str(parent_gui.current_mat_path.parent))
        else:
            self.folder_edit.setText(str(parent_gui.default_patient_folder))
        self.refresh_files()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.folder_edit = QtWidgets.QLineEdit(str(self.parent_gui.default_patient_folder))
        row.addWidget(QtWidgets.QLabel("Patient folder:")); row.addWidget(self.folder_edit, stretch=1)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_folder); row.addWidget(browse)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_files); row.addWidget(refresh)

        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.patient_combo = QtWidgets.QComboBox(); row.addWidget(QtWidgets.QLabel("Patient:")); row.addWidget(self.patient_combo, stretch=1)
        load = QtWidgets.QPushButton("Load selected")
        load.clicked.connect(self.load_selected); row.addWidget(load)
        prevb = QtWidgets.QPushButton("Previous")
        prevb.clicked.connect(lambda: self.step_patient(-1)); row.addWidget(prevb)
        nextb = QtWidgets.QPushButton("Next")
        nextb.clicked.connect(lambda: self.step_patient(1)); row.addWidget(nextb)
        save = QtWidgets.QPushButton("Save current source")
        save.clicked.connect(self.parent_gui.save_source_location); row.addWidget(save)

        info = QtWidgets.QLabel("Workflow: load each original patient, click/drag source on the main 3-plane viewer, save source, then run the sweep. Batch MCX uses each patient source unchanged while transforming fetal labels.")
        info.setWordWrap(True); layout.addWidget(info)

        grid = QtWidgets.QGridLayout(); layout.addLayout(grid)
        headers = ["from", "to", "step"]
        for j,h in enumerate(headers): grid.addWidget(QtWidgets.QLabel(h), 0, j+1)
        self.rows = {}
        defaults = {
            "Rx": (-10, 10, 5), "Ry": (-10, 10, 5), "Rz": (-10, 10, 5),
            "Tx": (-5, 5, 5), "Ty": (-5, 5, 5), "Tz": (-5, 5, 5),
            "Scale": (90, 110, 10),
        }
        for i,(name,vals) in enumerate(defaults.items(), start=1):
            grid.addWidget(QtWidgets.QLabel(name + (" (%)" if name=="Scale" else " (deg/mm)")), i, 0)
            spins=[]
            for j,val in enumerate(vals):
                sp=QtWidgets.QDoubleSpinBox(); sp.setRange(-1000, 1000); sp.setDecimals(3); sp.setValue(val); sp.setMaximumWidth(100)
                sp.valueChanged.connect(self.update_estimate)
                grid.addWidget(sp, i, j+1); spins.append(sp)
            self.rows[name]=spins

        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.mode_combo = QtWidgets.QComboBox(); self.mode_combo.addItems(["1D", "Grid"]); self.mode_combo.currentTextChanged.connect(self.update_estimate)
        row.addWidget(QtWidgets.QLabel("Sweep mode:")); row.addWidget(self.mode_combo)
        self.layers_combo = QtWidgets.QComboBox(); self.layers_combo.addItems(["3", "5", "7"]); self.layers_combo.setCurrentText(str(self.parent_gui.n_layers))
        row.addWidget(QtWidgets.QLabel("Layers:")); row.addWidget(self.layers_combo)
        self.wavelength_edit = QtWidgets.QLineEdit("780")
        row.addWidget(QtWidgets.QLabel("Wavelengths nm, comma-separated:")); row.addWidget(self.wavelength_edit)
        self.photon_spin = QtWidgets.QSpinBox(); self.photon_spin.setRange(1000, 1000000000); self.photon_spin.setValue(int(self.parent_gui.photon_count.value()))
        row.addWidget(QtWidgets.QLabel("Photons:")); row.addWidget(self.photon_spin)
        self.det_sep_spin = QtWidgets.QDoubleSpinBox(); self.det_sep_spin.setRange(1, 200); self.det_sep_spin.setValue(float(self.parent_gui.det_sep.value())); self.det_sep_spin.setSuffix(" mm")
        row.addWidget(QtWidgets.QLabel("S-D offset:")); row.addWidget(self.det_sep_spin)

        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.active_fetus_combo = QtWidgets.QComboBox(); self.active_fetus_combo.addItems(["Fetus 1 labels 110-134", "Twin labels 210-234"])
        row.addWidget(QtWidgets.QLabel("Transform active fetus:")); row.addWidget(self.active_fetus_combo)
        self.det_radius_spin = QtWidgets.QSpinBox(); self.det_radius_spin.setRange(1,100); self.det_radius_spin.setValue(10)
        row.addWidget(QtWidgets.QLabel("Detector radius vox:")); row.addWidget(self.det_radius_spin)
        self.run_mcx_check = QtWidgets.QCheckBox("Run PMCX after transform"); self.run_mcx_check.setChecked(True)
        row.addWidget(self.run_mcx_check)
        row.addStretch(1)

        row = QtWidgets.QHBoxLayout(); layout.addLayout(row)
        self.estimate_label = QtWidgets.QLabel("Estimated runs: --"); row.addWidget(self.estimate_label)
        row.addStretch(1)
        run = QtWidgets.QPushButton("Run batch transform + MCX sweep")
        run.clicked.connect(self.run_batch); row.addWidget(run)
        close = QtWidgets.QPushButton("Close"); close.clicked.connect(self.accept); row.addWidget(close)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); layout.addWidget(self.log, stretch=1)
        self.update_estimate()

    def browse_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder with Fetus_Model-*.mat", self.folder_edit.text())
        if d:
            self.folder_edit.setText(d); self.refresh_files()

    def refresh_files(self):
        self.files = discover_patient_files(self.folder_edit.text())
        self.patient_combo.clear()
        for f in self.files:
            self.patient_combo.addItem(f.name, str(f))
        self.log.appendPlainText(f"Found {len(self.files)} patient .mat files.")
        self.update_estimate()

    def load_selected(self):
        path = self.patient_combo.currentData()
        if path:
            self.parent_gui.load_mat_path(path)
            self.log.appendPlainText(f"Loaded {Path(path).name} in main viewer.")

    def step_patient(self, delta):
        n = self.patient_combo.count()
        if n == 0: return
        i = (self.patient_combo.currentIndex() + delta) % n
        self.patient_combo.setCurrentIndex(i)
        self.load_selected()

    def params(self):
        return {k: tuple(sp.value() for sp in v) for k,v in self.rows.items()}

    def wavelengths(self):
        vals=[]
        for part in self.wavelength_edit.text().replace(';', ',').split(','):
            part=part.strip()
            if part:
                vals.append(float(part))
        return vals or [780.0]

    def update_estimate(self):
        combos = generate_transform_combos(self.params(), self.mode_combo.currentText())
        n = len(combos) * max(1, len(self.files)) * len(self.wavelengths())
        self.estimate_label.setText(f"Estimated runs: {n}  ({len(self.files)} patients x {len(combos)} transforms x {len(self.wavelengths())} wavelengths)")

    def run_batch(self):
        try:
            import pmcx
        except ImportError:
            if self.run_mcx_check.isChecked():
                QtWidgets.QMessageBox.warning(self, "PMCX missing", "Install PMCX first: pip install pmcx")
                return
            pmcx = None
        self.refresh_files()
        if not self.files:
            QtWidgets.QMessageBox.warning(self, "No files", "No Fetus_Model-*.mat files found.")
            return
        store = self.parent_gui.read_source_store()
        combos = generate_transform_combos(self.params(), self.mode_combo.currentText())
        wavelengths = self.wavelengths()
        n_layers = int(self.layers_combo.currentText())
        active_fetus = 1 if self.active_fetus_combo.currentIndex() == 0 else 2
        root = Path(self.folder_edit.text()).expanduser() / "batch_mcx_results"
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = root / f"batch_log_{timestamp}.txt"
        total = len(self.files) * len(combos) * len(wavelengths)
        progress = QtWidgets.QProgressDialog("Running batch sweep...", "Cancel", 0, total, self)
        progress.setWindowTitle("Batch MCX sweep")
        progress.setMinimumDuration(0)
        done = 0
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"Batch sweep started {timestamp}\n")
            for f in self.files:
                if progress.wasCanceled(): break
                patient_id = self.parent_gui.infer_patient_id(f)
                if patient_id not in store:
                    msg = f"SKIP {patient_id}: no saved source in {self.parent_gui.source_store_path()}\n"
                    self.log.appendPlainText(msg.strip()); log.write(msg); continue
                source_ct = store[patient_id]["source_ct"]
                try:
                    vol_ct, voxel_size_mm, _ = load_mat_v73_volume(f)
                except Exception as e:
                    msg = f"ERROR loading {f.name}: {e}\n"; self.log.appendPlainText(msg.strip()); log.write(msg); continue
                base, f_sub, f_ids, ctr_mm, r_frame = precompute_fetus_transform(vol_ct, voxel_size_mm, active_fetus)
                for combo in combos:
                    if progress.wasCanceled(): break
                    tid = transform_id(combo)
                    transformed = apply_fetus_transform(base, f_sub, f_ids, voxel_size_mm, ctr_mm, r_frame, combo)
                    for wav in wavelengths:
                        if progress.wasCanceled(): break
                        patient_folder = root / f"{patient_id}_{int(wav) if wav.is_integer() else wav:g}nm_{n_layers}layers"
                        outdir = patient_folder / tid
                        outdir.mkdir(parents=True, exist_ok=True)
                        meta = {
                            "patient_id": patient_id,
                            "source_ct": source_ct,
                            "source_mcx_1_based": ct_to_mcx_pos(*source_ct),
                            "transform_combo": {"Rx_deg": combo[0], "Ry_deg": combo[1], "Rz_deg": combo[2], "Tx_mm": combo[3], "Ty_mm": combo[4], "Tz_mm": combo[5], "Scale_percent": combo[6]},
                            "wavelength_nm": wav,
                            "layers": n_layers,
                            "voxel_size_mm": voxel_size_mm.tolist(),
                            "original_mat_file": str(f),
                            "coordinate_note": "CT [x,y,z] -> MCX [x,z,y]; MCX z is CT y depth. Source is kept at the original patient location.",
                        }
                        np.save(outdir / "transformed_volume_ct.npy", transformed)
                        with open(outdir / "metadata.json", "w", encoding="utf-8") as mf: json.dump(meta, mf, indent=2)
                        if self.run_mcx_check.isChecked():
                            cfg = build_batch_cfg(transformed, voxel_size_mm, n_layers, wav, source_ct, self.photon_spin.value(), self.det_sep_spin.value(), self.det_radius_spin.value())
                            result = pmcx.mcxlab(cfg)
                            np.savez_compressed(outdir / "pmcx_result.npz", **result)
                            vol_mcx = cfg["vol"]
                            np.save(outdir / "mcx_volume_layers.npy", vol_mcx)
                        msg = f"DONE {patient_id} {wav:g}nm {n_layers}layers {tid}"
                        self.log.appendPlainText(msg); log.write(msg + "\n"); log.flush()
                        done += 1; progress.setValue(done); QtWidgets.QApplication.processEvents()
        progress.close()
        QtWidgets.QMessageBox.information(self, "Batch complete", f"Completed {done} outputs.\nLog: {log_path}")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    gui = FetusMCXGui()
    gui.show()
    app.exec_()
