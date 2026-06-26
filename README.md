# STORX Fetus MCX GUI

Python GUI for visualizing segmented pregnant-patient CT volumes, assigning optical tissue layers, selecting patient-specific source locations, running MCX/PMCX photon-transport simulations, and reviewing batch Monte Carlo results.

> **Data note:** This repository includes converted `.mat` fetal CT label volumes in the `volumes/` folder. These files were converted from the openly available source dataset for convenience with the Python GUI. Generated MCX/PMCX outputs, transformed volumes, and batch result folders should not be committed.. Do not commit patient CT volumes, transformed volumes, or Monte Carlo result files unless your team has explicit approval to share them. The included `.gitignore` excludes `volumes/`, `mcx_export/`, `batch_mcx_results/`, `.mat`, `.npy`, `.npz`, and `.bin` files by default.

## Main features

- Loads original CT volumes from a `volumes/` subfolder on startup.
- Supports patient navigation with previous/next controls.
- Displays three orthogonal CT planes.
- Converts segmented CT labels into simplified optical tissue models with 3, 5, or 7 layers.
- Lets the user click or drag a point source location near the maternal belly surface.
- Saves and reloads patient-specific source locations using `mcx_export/source_locations.json`.
- Converts CT coordinates to MCX coordinates so photon depth follows CT Y:

```text
CT  [x, y, z]
MCX [x, z, y]
```

- Runs single-patient PMCX simulations.
- Runs batch transformations and batch Monte Carlo sweeps.
- Loads a folder of MCX results and browses them using previous/next controls.
- Overlays photon fluence/flux on the transformed tissue geometry.
- Supports boundaries-only visualization with boundary colors matching tissue IDs.

## Expected folder structure

Place the GUI script in your project folder and put all original patient `.mat` CT volumes in a subfolder named `volumes`.

```text
STORX_Python/
├── fetus_mcx_gui.py
├── requirements.txt
├── volumes/
│   ├── Fetus_Model-1.mat
│   ├── Fetus_Model-2.mat
│   ├── Fetus_Model-3.mat
│   └── ...
└── mcx_export/
    └── source_locations.json
```

The `mcx_export/` folder is created automatically when source locations or MCX files are saved.

## Installation

Create and activate a Python environment. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If your system uses the Windows Store Python path, you can also run the script with the full Python executable path, but a virtual environment is recommended.

## Running the GUI

```powershell
python fetus_mcx_gui.py
```

The GUI scans `./volumes/` on startup. If valid `.mat` files are found, the first patient loads automatically.

## Selecting and saving source locations

1. Use the patient dropdown or previous/next buttons to choose a patient.
2. Click near the maternal belly surface in any plane to place the source.
3. Drag the source marker to fine tune the location.
4. Click **Save source**.
5. Move to the next patient and repeat.

Saved source locations are stored in:

```text
mcx_export/source_locations.json
```

When the GUI is closed and reopened, the saved source location is automatically restored when that patient is loaded.

## Optical layer models

The GUI supports simplified optical models with 3, 5, or 7 tissue layers.

Example 7-layer model:

| ID | Layer |
|---:|---|
| 1 | Maternal skin |
| 2 | Maternal adipose |
| 3 | Maternal muscle |
| 4 | Uterine wall |
| 5 | Amniotic fluid |
| 6 | Placenta |
| 7 | Fetus |

Rows in the optical-property table are color coded to match the displayed tissue layers. Boundary overlays use the same ID colors.

## Running a single MCX simulation

1. Load a patient.
2. Select or confirm the source location.
3. Choose the desired number of layers.
4. Set photon count and detector settings.
5. Click **Run PMCX**.

After the run, use:

- **Show photons/fluence** to overlay the photon fluence/flux.
- **Boundaries only** to hide filled tissue regions and show colored boundaries.

## Batch workflow

The intended batch workflow is:

1. Load original patient volumes from `volumes/`.
2. Go through each original patient once and save the source location.
3. Open the batch source/sweep dialog.
4. Select transformation limits and step sizes.
5. Run batch transformation and Monte Carlo sweeps.
6. Save each result using a patient/wavelength/layer/transform folder structure.

Typical result folders are organized like:

```text
batch_mcx_results/
└── fetus-1_780nm_7layers/
    └── Rx0_Ry0_Rz0_Tx0_Ty0_Tz0_S100/
        ├── transformed_volume_ct.npy
        ├── mcx_volume_layers.npy
        ├── pmcx_result.npz
        └── metadata.json
```

## Loading batch MCX results for visualization

Click **Load MCX result folder**.

You can select either:

- the root batch folder, such as `batch_mcx_results/`, or
- a specific transformed-result folder containing `pmcx_result.npz`.

If the root folder is selected, the GUI recursively indexes all result folders. It then loads one result at a time so large photon volumes are not all loaded into memory at once.

Use:

- **Prev result**
- **Next result**
- result dropdown

The GUI displays the patient ID, wavelength, layer number, transformation parameters, and current result index.

## Coordinate convention

The original CT volume is loaded as:

```text
vol_ct[x, y, z]
```

where CT Z runs approximately neck-to-legs. For MCX, photon depth should be into tissue, so the GUI converts to:

```text
vol_mcx[x, z, y]
```

Therefore:

| Display axis | Meaning |
|---|---|
| MCX X | CT X |
| MCX Y | CT Z, neck-to-legs |
| MCX Z | CT Y, tissue depth |

When viewing photon transport, the most important depth view is usually the MCX XZ plane, corresponding to CT X versus CT Y depth.

## Wavelengths and optical properties

The current GUI is structured for wavelength sweeps. Confirm that the optical property table for each wavelength is populated before using non-780 nm simulations for final results.

Recommended future input format:

```text
wavelength_nm,layer_id,layer_name,mua_mm^-1,mus_mm^-1,g,n
780,1,Maternal skin,0.0406,14.3840,0.8,1.4000
780,2,Maternal adipose,0.0022,13.6470,0.9,1.3700
...
```

## Troubleshooting

### No volumes found

Make sure your original `.mat` files are inside:

```text
./volumes/
```

### Source point does not reload

Check that this file exists:

```text
mcx_export/source_locations.json
```

Also confirm that the patient ID inferred from the file name matches the saved key, such as `fetus-1`.

### PMCX not installed

Install PMCX inside the same Python environment:

```powershell
pip install pmcx
```

### `detected 0 photons`

The simulation may still have computed internal fluence. Zero detected photons only means no photons reached the detector sphere. For debugging, reduce the source-detector separation, increase detector radius, use fewer layers first, or increase the photon count.

### GUI too wide for monitor

Some Windows displays may print `QWindowsWindow::setGeometry` warnings if the toolbar is wider than the monitor. This is usually a warning, not a crash. A future layout can move controls into a side panel if needed.

## Suggested repository contents

```text
fetus_mcx_gui.py
requirements.txt
README.md
.gitignore
```

Do not commit raw CT volumes or MCX output data unless approved.
