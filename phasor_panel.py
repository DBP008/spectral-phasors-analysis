import io

import panel as pn
import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv
from holoviews import streams
from itertools import cycle
from matplotlib.path import Path as MplPath
import colorcet
import tifffile
import hvplot.xarray   # noqa: F401 – registers hvplot accessor
import hvplot.pandas   # noqa: F401 – registers hvplot accessor on DataFrames

pn.extension("bokeh", sizing_mode="stretch_width")
hv.extension("bokeh")

# ── Core math helpers ────────────────────────────────────────────────────────


def calculate_phasor_transform(ds: xr.Dataset) -> xr.Dataset:
    fft_values = xr.apply_ufunc(
        np.fft.fft,
        ds.intensity,
        input_core_dims=[["wavelength"]],
        output_core_dims=[["harmonic_bin"]],
    )
    dc_component = fft_values.isel(harmonic_bin=0).real
    h = int(ds.attrs.get("harmonic", 1))
    harmonic_data = fft_values.isel(harmonic_bin=h)
    ds["G"] = harmonic_data.real / dc_component
    ds["S"] = harmonic_data.imag / dc_component
    return ds


def add_spectral_reference(ds: xr.Dataset, n_ref_points: int = 1000) -> xr.Dataset:
    harmonic = int(ds.attrs.get("harmonic", 1))
    w_min = ds["wavelength"].min().item()
    w_max = ds["wavelength"].max().item()
    w_ref = np.linspace(w_min, w_max, n_ref_points)
    indices = np.arange(n_ref_points)
    phase = -2 * np.pi * harmonic * indices / n_ref_points
    ds = ds.assign_coords(wavelength_ref=w_ref)
    ds["G_ref"] = (("wavelength_ref",), np.cos(phase))
    ds["S_ref"] = (("wavelength_ref",), np.sin(phase))
    return ds


def make_gaussian(wavelengths: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.exp(-0.5 * ((wavelengths - mean) / std) ** 2)


def make_spectra_dataset(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    harmonic_n: int,
    sample_names: list[str] | None = None,
) -> xr.Dataset:
    """Build a clean xr.Dataset from a (samples × wavelength) array.

    If *sample_names* is provided, a ``sample_name`` coordinate is attached
    along the ``sample`` dimension.  The ``sample`` dimension itself stays
    integer-valued so positional indexing keeps working everywhere.
    """
    n_samples = intensities.shape[0]
    coords: dict = {
        "sample": np.arange(n_samples),
        "wavelength": wavelengths,
    }
    if sample_names is not None:
        coords["sample_name"] = ("sample", list(sample_names))

    ds = xr.Dataset(
        data_vars={
            "intensity": (["sample", "wavelength"], intensities),
            "G": (["sample"], np.full(n_samples, np.nan)),
            "S": (["sample"], np.full(n_samples, np.nan)),
        },
        coords=coords,
        attrs={"harmonic": harmonic_n},
    )
    ds.wavelength.attrs = {"units": "nm"}
    ds.G.attrs = {"long_name": "Phasor G (real)"}
    ds.S.attrs = {"long_name": "Phasor S (imaginary)"}
    return ds


def get_wavelengths(start: float, step: float, end: float) -> np.ndarray:
    return np.arange(start, end + step / 2, step)


def parse_uploaded_csv(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse an instrument-exported CSV into (wavelengths, intensities, names).

    Expected layout
    ---------------
    Row 0 : sample names in columns 0, 2, 4, …  (odd cols empty)
    Row 1 : repeated "Wavelength (nm), Intensity (a.u.)" headers
    Row 2+ : data pairs  (wavelength, intensity) × N_samples
    A blank row marks the end of the spectral data.

    Returns
    -------
    wavelengths : 1-D array  (N_wl,)
    intensities : 2-D array  (N_samples, N_wl)
    names       : list[str]  length N_samples
    """
    text = file_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # ── Row 0: sample names ──────────────────────────────────────────────
    header_cells = lines[0].split(",")
    # Names sit at even indices (0, 2, 4, …)
    names = [header_cells[i].strip() for i in range(0, len(header_cells), 2) if header_cells[i].strip()]

    # ── Find the first blank row (signals end of data) ───────────────────
    data_end = len(lines)
    for idx, line in enumerate(lines[2:], start=2):   # skip header rows
        if line.strip() == "" or line.replace(",", "").strip() == "":
            data_end = idx
            break

    # ── Parse numeric block ──────────────────────────────────────────────
    # Use pandas for robust float parsing of the data block
    csv_block = "\n".join(lines[2:data_end])
    df = pd.read_csv(io.StringIO(csv_block), header=None)

    n_samples = len(names)
    wavelengths = df.iloc[:, 0].to_numpy(dtype=float)
    intensities = np.column_stack(
        [df.iloc[:, 2 * i + 1].to_numpy(dtype=float) for i in range(n_samples)]
    ).T  # shape (n_samples, n_wavelengths)

    return wavelengths, intensities, names


# ── Plot helpers ─────────────────────────────────────────────────────────────


def create_phasor_plot(ds: xr.Dataset):
    df_samples = ds[["G", "S"]].to_dataframe().reset_index()
    df_ref = ds[["G_ref", "S_ref"]].to_dataframe().reset_index()

    ref_chart = df_ref.hvplot.points(
        x="G_ref",
        y="S_ref",
        color="wavelength_ref",
        cmap="spectral_r",
        size=28,
        colorbar=False,
        tools=["hover"],
    )

    single_sample = len(df_samples) == 1
    if single_sample:
        color_mapping = {int(df_samples["sample"].iloc[0]): "steelblue"}
        sample_chart = df_samples.hvplot.points(
            x="G",
            y="S",
            color="steelblue",
            size=50,
            colorbar=False,
            tools=["hover", "box_select", "lasso_select", "tap"],
        )
    else:
        color_mapping = dict(
            zip(df_samples["sample"].tolist(), cycle(colorcet.b_glasbey_hv))
        )
        sample_chart = df_samples.hvplot.points(
            x="G",
            y="S",
            color="sample",
            cmap=color_mapping,
            size=45,
            colorbar=False,
            tools=["hover", "box_select", "lasso_select", "tap"],
        )

    phasor_plot = (ref_chart * sample_chart).opts(
        hv.opts.Points(
            frame_width=500,
            frame_height=500,
            padding=0.1,
            xlabel="G",
            ylabel="S",
            show_grid=True,
            show_legend=False,
            title="Phasor Plot",
        )
    )
    return phasor_plot, sample_chart, color_mapping


def build_spectrum_dmap(
    ds: xr.Dataset,
    sample_plot: hv.Points,
    color_mapping: dict,
    show_individual: bool = False,
) -> hv.DynamicMap:
    """Return a DynamicMap wired to *sample_plot*'s Selection1D stream.

    When show_individual is True, individual spectra are rendered as a
    pre-built hv.NdOverlay using the same *color_mapping* dict as the
    phasor plot so dot and line colours correspond 1-to-1.
    Selection filters the overlay with .select(sample=index).
    """
    selection = streams.Selection1D(source=sample_plot)

    # Build the full individual-spectra overlay once (outside the callback)
    # so filtering is cheap and colours are consistent.
    hv_ds = hv.Dataset(
        ds.intensity.to_dataframe().reset_index(),
        kdims=["wavelength", "sample"],
        vdims="intensity",
    )
    individual_plot = hv_ds.to(hv.Curve, "wavelength", "intensity").overlay("sample").opts(
        hv.opts.Curve(
            color=hv.Cycle(list(color_mapping.values())),
            frame_width=800,
            frame_height=500,
            line_width=3,
            tools=["hover"],
            # hover_tooltips=[("sample","@sample"), ("wavelength","@x"), ("intensity","@y")],
            xlabel="wavelength (nm)",
            ylabel="intensity",
            title="spectrum"
        ),
        hv.opts.NdOverlay(show_legend=False, batched=True),
    )

    def select_spectrum(index):
        if show_individual and len(index) <= 300:
            if not index:
                return individual_plot
            return individual_plot.select(sample=index)

        # ── Average mode ────────────────────────────────────────────────────
        if not index:
            selected = ds.intensity
            title = "All Samples"
        else:
            selected = ds.intensity.isel(sample=index)
            title = f"{len(index)} Selected"

        suffix = " [avg mode: >300]" if show_individual else ""
        avg_curve = selected.mean("sample", keep_attrs=True).hvplot.line(
            x="wavelength",
            y="intensity",
            title=f"Average Spectrum ({title}){suffix}",
            color="black",
            line_width=3,
            frame_width=800,
            frame_height=500,
        )
        return hv.NdOverlay({"Average": avg_curve}, kdims="sample").opts(
            show_legend=False,
        )

    return hv.DynamicMap(select_spectrum, streams=[selection]).opts(framewise=True)


# ── Global widgets ────────────────────────────────────────────────────────────

start_lambda_input = pn.widgets.FloatInput(
    name="Start λ (nm)", value=400, step=10, width=160
)
step_lambda_input = pn.widgets.FloatInput(
    name="Step λ (nm)", value=5, step=1, width=160
)
end_lambda_input = pn.widgets.FloatInput(
    name="End λ (nm)", value=700, step=10, width=160
)
harmonic_input = pn.widgets.IntInput(
    name="Harmonic N", value=1, step=1, start=1, width=160
)
show_individual = pn.widgets.Toggle(
    name="Show Individual Spectra", button_type="default", value=False
)

# ── Tab 1 widgets: single gaussian ───────────────────────────────────────────

t1_mean = pn.widgets.FloatSlider(
    name="Mean (nm)", start=400, end=700, value=550, step=1
)
t1_std = pn.widgets.FloatSlider(
    name="Std (nm)", start=1, end=150, value=20, step=1
)

# ── Tab 3 widgets: CSV upload ─────────────────────────────────────────────────

t3_file_input = pn.widgets.FileInput(accept=".csv", multiple=False, name="Upload CSV")

# ── Tab 4 widgets: TIFF image upload ─────────────────────────────────────────

t4_file_input = pn.widgets.FileInput(accept=".tif,.tiff", multiple=False, name="Upload TIFF")

# ── Tab 2 widgets: multi gaussian + noise ────────────────────────────────────

t2_n_samples = pn.widgets.IntSlider(
    name="N samples", start=10, end=500, value=100, step=10
)
t2_mean_base = pn.widgets.FloatSlider(
    name="Center wavelength λ (nm)", start=400, end=700, value=550, step=1
)
t2_mean_spread = pn.widgets.FloatSlider(
    name="Wavelength variance σ (nm)", start=0, end=100, value=10, step=1
)
t2_std_base = pn.widgets.FloatSlider(
    name="Wavelength jitter Δλ (nm)", start=1, end=150, value=30, step=1
)
t2_std_spread = pn.widgets.FloatSlider(
    name="Variance jitter Δσ (nm)", start=0, end=60, value=5, step=1
)
t2_add_noise = pn.widgets.Toggle(
    name="Add Gaussian Noise", button_type="default", value=False
)
t2_snr = pn.widgets.FloatSlider(
    name="SNR (dB)", start=0, end=40, value=20, step=1, disabled=True
)

def _toggle_snr(event):
    t2_snr.disabled = not event.new

t2_add_noise.param.watch(_toggle_snr, "value")


# ── View callbacks ────────────────────────────────────────────────────────────


def _validate_wavelengths(start, step, end):
    if start >= end:
        return pn.pane.Alert("Start λ must be less than End λ.", alert_type="danger")
    if step <= 0:
        return pn.pane.Alert("Step λ must be positive.", alert_type="danger")
    return None


@pn.depends(
    start_lambda_input,
    step_lambda_input,
    end_lambda_input,
    harmonic_input,
    t1_mean,
    t1_std,
    show_individual,
)
def tab1_view(start, step, end, h, mean, std, show_ind):
    err = _validate_wavelengths(start, step, end)
    if err:
        return err

    wl = get_wavelengths(start, step, end)
    # Single sample: shape (1, N_wl)
    intensity = make_gaussian(wl, mean, std)[np.newaxis, :]
    ds = make_spectra_dataset(wl, intensity, h)
    ds = calculate_phasor_transform(ds)
    ds = add_spectral_reference(ds)

    phasor_plot, sample_plot, color_mapping = create_phasor_plot(ds)
    dmap = build_spectrum_dmap(ds, sample_plot, color_mapping, show_individual=show_ind)
    return pn.Row(phasor_plot, dmap, sizing_mode="stretch_width")


@pn.depends(
    start_lambda_input,
    step_lambda_input,
    end_lambda_input,
    harmonic_input,
    t2_n_samples,
    t2_mean_base,
    t2_mean_spread,
    t2_std_base,
    t2_std_spread,
    t2_add_noise,
    t2_snr,
    show_individual,
)
def tab2_view(
    start, step, end, h,
    n_samples, mean_base, mean_spread, std_base, std_spread,
    add_noise, snr_db, show_ind,
):
    err = _validate_wavelengths(start, step, end)
    if err:
        return err

    rng = np.random.default_rng()
    wl = get_wavelengths(start, step, end)

    # Draw random means and stds for each sample
    means = (
        rng.normal(mean_base, mean_spread, n_samples)
        if mean_spread > 0
        else np.full(n_samples, float(mean_base))
    )
    stds = (
        np.clip(rng.normal(std_base, std_spread, n_samples), 0.5, None)
        if std_spread > 0
        else np.full(n_samples, float(std_base))
    )

    intensities = np.stack([make_gaussian(wl, m, s) for m, s in zip(means, stds)])

    if add_noise:
        snr_linear = 10 ** (snr_db / 10)
        # Per-sample signal power → noise std
        sig_power = np.mean(intensities ** 2, axis=1, keepdims=True)
        noise_std = np.sqrt(sig_power / snr_linear)
        noise = rng.standard_normal(intensities.shape) * noise_std
        intensities = np.clip(intensities + noise, 0, None)

    ds = make_spectra_dataset(wl, intensities, h)
    ds = calculate_phasor_transform(ds)
    ds = add_spectral_reference(ds)

    phasor_plot, sample_plot, color_mapping = create_phasor_plot(ds)
    dmap = build_spectrum_dmap(ds, sample_plot, color_mapping, show_individual=show_ind)
    return pn.Row(phasor_plot, dmap, sizing_mode="stretch_width")


@pn.depends(
    t3_file_input,
    harmonic_input,
    show_individual,
)
def tab3_view(file_bytes, h, show_ind):
    if file_bytes is None:
        return pn.pane.Alert(
            "Upload a CSV file to begin.  "
            "The wavelength range and spacing will be read from the file "
            "(the global λ parameters are **ignored** for this tab).",
            alert_type="warning",
        )

    try:
        wavelengths, intensities, sample_names = parse_uploaded_csv(file_bytes)
    except Exception as exc:
        return pn.pane.Alert(f"Error parsing CSV: {exc}", alert_type="danger")

    wl_start, wl_end = wavelengths[0], wavelengths[-1]
    wl_step = np.median(np.diff(wavelengths))
    sample_lines = "\n".join(
        f"| {i} | {name} |" for i, name in enumerate(sample_names)
    )
    info_md = pn.pane.Markdown(
        f"**CSV info** — {len(sample_names)} samples, \n\n"
        f"λ {wl_start:.1f}–{wl_end:.1f} nm, "
        f"step ≈ {wl_step:.2f} nm\n\n"
        f"| Index | Sample name |\n"
        f"|------:|-------------|\n"
        f"{sample_lines}"
    )
    warning = pn.pane.Alert(
        "⚠️ Wavelength range and spacing are taken from the uploaded CSV.  "
        "The global λ start / step / end parameters do **not** apply to this tab.",
        alert_type="warning",
    )

    ds = make_spectra_dataset(wavelengths, intensities, h, sample_names=sample_names)
    ds = calculate_phasor_transform(ds)
    ds = add_spectral_reference(ds)

    phasor_plot, sample_plot, color_mapping = create_phasor_plot(ds)
    dmap = build_spectrum_dmap(ds, sample_plot, color_mapping, show_individual=show_ind)
    return pn.Column(
        warning,
        info_md,
        pn.Row(phasor_plot, dmap, sizing_mode="stretch_width"),
    )


@pn.depends(
    t4_file_input,
    harmonic_input,
    start_lambda_input,
    end_lambda_input,
)
def tab4_view(file_bytes, h, start, end):
    combined_warning = pn.pane.Alert(
        "⚠️ **Show Individual Spectra** is not available in TIFF image mode — "
        "selections show averaged spectra only.  "
        "**Step λ** is ignored: wavelengths are computed as "
        "linspace(Start λ, End λ, n_frames) using the number of TIFF frames.",
        alert_type="warning",
    )

    if file_bytes is None:
        return pn.Column(
            combined_warning,
            pn.pane.Alert(
                "Upload a TIFF stack to begin.  "
                "Set Start λ and End λ in the Global Parameters sidebar first.",
                alert_type="secondary",
            ),
        )

    if start >= end:
        return pn.pane.Alert("Start λ must be less than End λ.", alert_type="danger")

    try:
        arr = tifffile.imread(io.BytesIO(file_bytes))
    except Exception as exc:
        return pn.pane.Alert(f"Error reading TIFF: {exc}", alert_type="danger")

    # Normalise to (n_frames, H, W)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        pass  # already (n_frames, H, W) from tifffile
    else:
        return pn.pane.Alert(
            f"Unexpected TIFF shape {arr.shape}. Expected a 3-D stack.",
            alert_type="danger",
        )

    n_frames, H, W = arr.shape
    wavelengths = np.linspace(start, end, n_frames)
    wl_step_auto = (end - start) / max(n_frames - 1, 1)

    info_md = pn.pane.Markdown(
        f"**TIFF info** — {n_frames} frames, {H}×{W} px  \n"
        f"λ {start:.1f}–{end:.1f} nm, auto step ≈ {wl_step_auto:.2f} nm"
    )

    # ── Build xarray Dataset ───────────────────────────────────────────────────
    intensities_2d = arr.astype(float).transpose(1, 2, 0).reshape(H * W, n_frames)
    x_coords = np.repeat(np.arange(H), W)
    y_coords = np.tile(np.arange(W), H)
    ds = xr.Dataset(
        data_vars={
            "intensity": (["sample", "wavelength"], intensities_2d),
            "G": (["sample"], np.full(H * W, np.nan)),
            "S": (["sample"], np.full(H * W, np.nan)),
        },
        coords={
            "sample": np.arange(H * W),
            "x": ("sample", x_coords),
            "y": ("sample", y_coords),
            "wavelength": wavelengths,
        },
        attrs={"harmonic": h},
    )
    ds.wavelength.attrs = {"units": "nm"}
    ds = calculate_phasor_transform(ds)
    ds = add_spectral_reference(ds)

    # ── Pre-extract arrays ─────────────────────────────────────────────────────
    _x      = ds["x"].values.astype(int)
    _y      = ds["y"].values.astype(int)
    _G      = ds["G"].values
    _S      = ds["S"].values
    _wl     = wavelengths
    _Gref   = ds["G_ref"].values
    _Sref   = ds["S_ref"].values
    _wlref  = ds["wavelength_ref"].values
    _intmat = ds["intensity"].values
    _xdim   = np.arange(H)
    _ydim   = np.arange(W)
    _EMPTY  = (0.0, 0.0, 0.0, 0.0)
    _IMG_WW = 420

    ds_indexed = ds.set_index(sample=["x", "y"])
    intensity_unstacked = ds_indexed["intensity"].unstack("sample")  # (wavelength, x, y)

    # ── Mask helpers ───────────────────────────────────────────────────────────
    def _imask_box(b):
        if b is None or b == _EMPTY:
            return np.zeros(len(_x), dtype=bool)
        x0, y0, x1, y1 = b
        return ((_x >= min(x0, x1)) & (_x <= max(x0, x1)) &
                (_y >= min(y0, y1)) & (_y <= max(y0, y1)))

    def _pmask_box(b):
        if b is None or b == _EMPTY:
            return np.zeros(len(_G), dtype=bool)
        g0, s0, g1, s1 = b
        return ((_G >= min(g0, g1)) & (_G <= max(g0, g1)) &
                (_S >= min(s0, s1)) & (_S <= max(s0, s1)))

    def _imask_lasso(geom):
        if geom is None or len(geom) < 3:
            return np.zeros(len(_x), dtype=bool)
        return MplPath(geom).contains_points(np.column_stack([_x.astype(float), _y.astype(float)]))

    def _pmask_lasso(geom):
        if geom is None or len(geom) < 3:
            return np.zeros(len(_G), dtype=bool)
        return MplPath(geom).contains_points(np.column_stack([_G, _S]))

    def _imask(bounds, geom):
        return _imask_box(bounds) | _imask_lasso(geom)

    def _pmask(bounds, geom):
        return _pmask_box(bounds) | _pmask_lasso(geom)

    # ── Controls ───────────────────────────────────────────────────────────────
    wl_slider_t4 = pn.widgets.DiscreteSlider(
        name="Wavelength (nm)",
        options={f"{w:.1f} nm": i for i, w in enumerate(_wl)},
        value=0,
        width=_IMG_WW,
    )
    mode_toggle_t4 = pn.widgets.RadioButtonGroup(
        name="Selection mode",
        options=["Image", "Phasor"],
        value="Image",
        button_type="primary",
    )
    reset_btn_t4 = pn.widgets.Button(name="↺  Reset", button_type="warning", width=100)

    # ── Streams ────────────────────────────────────────────────────────────────
    img_bounds_t4 = streams.BoundsXY(bounds=_EMPTY)
    ph_bounds_t4  = streams.BoundsXY(bounds=_EMPTY)
    img_lasso_t4  = streams.Lasso(geometry=None)
    ph_lasso_t4   = streams.Lasso(geometry=None)

    def _reset_all_t4():
        img_bounds_t4.event(bounds=_EMPTY)
        ph_bounds_t4.event(bounds=_EMPTY)
        img_lasso_t4.event(geometry=None)
        ph_lasso_t4.event(geometry=None)

    mode_toggle_t4.param.watch(lambda e: _reset_all_t4(), "value")
    reset_btn_t4.on_click(lambda e: _reset_all_t4())

    # ── Base image DynamicMap ──────────────────────────────────────────────────
    def _base_image_t4(wl_idx, mode, ibounds, igeom):
        data  = intensity_unstacked.isel(wavelength=wl_idx).values
        has_img_sel = _imask(ibounds, igeom).any()
        go_gray = (mode == "Phasor") or (mode == "Image" and has_img_sel)
        cmap  = "gray"  if go_gray else "viridis"
        alpha = 0.25    if go_gray else 1.0
        return hv.Image(
            (_xdim, _ydim, data.T),
            kdims=["x", "y"], vdims=["intensity"],
        ).opts(
            cmap=cmap, colorbar=True, alpha=alpha,
            frame_width=_IMG_WW, frame_height=_IMG_WW,
            title=f"Image  —  {_wl[wl_idx]:.1f} nm",
            tools=["box_select", "lasso_select", "wheel_zoom"],
            active_tools=["wheel_zoom"],
        )

    base_img_dmap_t4 = hv.DynamicMap(pn.bind(
        _base_image_t4,
        wl_idx=wl_slider_t4,
        mode=mode_toggle_t4,
        ibounds=img_bounds_t4.param.bounds,
        igeom=img_lasso_t4.param.geometry,
    ))
    img_bounds_t4.source = base_img_dmap_t4
    img_lasso_t4.source  = base_img_dmap_t4

    # ── Static phasor background ───────────────────────────────────────────────
    phasor_bg_t4 = hv.Points(pd.DataFrame({"G": _G, "S": _S}), kdims=["G", "S"]).opts(
        color="lightgray", size=3, alpha=0.3,
        frame_width=_IMG_WW, frame_height=_IMG_WW,
        tools=["box_select", "lasso_select", "wheel_zoom"],
        active_tools=["wheel_zoom"],
        xlabel="G", ylabel="S", show_grid=True, padding=0.1,
        title="Phasor",
    )
    ph_bounds_t4.source = phasor_bg_t4
    ph_lasso_t4.source  = phasor_bg_t4

    # ── Reference arc ─────────────────────────────────────────────────────────
    ref_arc_t4 = hv.Points(
        pd.DataFrame({"G": _Gref, "S": _Sref, "wl": _wlref}),
        kdims=["G", "S"], vdims=["wl"],
    ).opts(color="wl", cmap="Spectral_r", size=6, alpha=0.6, colorbar=False, tools=["hover"])

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _make_phasor_overlay_t4(ibounds, igeom, pbounds, pgeom, mode):
        if mode == "Image":
            m, color = _imask(ibounds, igeom), "steelblue"
        else:
            m, color = _pmask(pbounds, pgeom), "tomato"
        df = pd.DataFrame({"G": _G[m], "S": _S[m]}) if m.any() else pd.DataFrame({"G": [], "S": []})
        return hv.Points(df, kdims=["G", "S"]).opts(color=color, size=5, alpha=0.9)

    def _make_image_overlay_t4(ibounds, igeom, pbounds, pgeom, wl_idx, mode):
        data = intensity_unstacked.isel(wavelength=wl_idx).values.copy()
        if mode == "Image":
            m = _imask(ibounds, igeom)
        else:
            m = _pmask(pbounds, pgeom)
        if not m.any():
            empty = np.full((H, W), np.nan)
            return hv.Image((_xdim, _ydim, empty), kdims=["x", "y"], vdims=["intensity"]).opts(
                alpha=0, colorbar=False,
            )
        sel_grid = np.zeros((H, W), dtype=bool)
        sel_grid[_x[m], _y[m]] = True
        masked = np.where(sel_grid, data, np.nan)
        return hv.Image(
            (_xdim, _ydim, masked.T),
            kdims=["x", "y"], vdims=["intensity"],
        ).opts(
            cmap="viridis", alpha=1.0, colorbar=False,
            clim=(float(np.nanmin(data)), float(np.nanmax(data))),
        )

    def _make_spectrum_t4(ibounds, igeom, pbounds, pgeom, mode):
        if mode == "Image":
            mask, color, src = _imask(ibounds, igeom), "steelblue", "image sel"
        else:
            mask, color, src = _pmask(pbounds, pgeom), "tomato", "phasor sel"
        n = int(mask.sum())
        if n == 0:
            avg   = _intmat.mean(axis=0)
            title = "Average Spectrum (all pixels)"
            color = "gray"
        else:
            avg   = _intmat[mask].mean(axis=0)
            title = f"Average Spectrum — {src}  ({n} px)"
        return hv.Curve(
            pd.DataFrame({"wavelength": _wl, "intensity": avg}),
            kdims=["wavelength"], vdims=["intensity"],
        ).opts(
            frame_width=560, frame_height=_IMG_WW,
            title=title, color=color,
            xlabel="Wavelength (nm)", ylabel="Intensity",
            line_width=2, show_grid=True, tools=["hover"],
        )

    # ── Wire DynamicMaps ───────────────────────────────────────────────────────
    img_overlay_dmap_t4 = hv.DynamicMap(pn.bind(
        _make_image_overlay_t4,
        ibounds=img_bounds_t4.param.bounds,
        igeom=img_lasso_t4.param.geometry,
        pbounds=ph_bounds_t4.param.bounds,
        pgeom=ph_lasso_t4.param.geometry,
        wl_idx=wl_slider_t4,
        mode=mode_toggle_t4,
    ))
    ph_overlay_dmap_t4 = hv.DynamicMap(pn.bind(
        _make_phasor_overlay_t4,
        ibounds=img_bounds_t4.param.bounds,
        igeom=img_lasso_t4.param.geometry,
        pbounds=ph_bounds_t4.param.bounds,
        pgeom=ph_lasso_t4.param.geometry,
        mode=mode_toggle_t4,
    ))
    spectrum_t4 = hv.DynamicMap(pn.bind(
        _make_spectrum_t4,
        ibounds=img_bounds_t4.param.bounds,
        igeom=img_lasso_t4.param.geometry,
        pbounds=ph_bounds_t4.param.bounds,
        pgeom=ph_lasso_t4.param.geometry,
        mode=mode_toggle_t4,
    )).opts(framewise=True)

    # ── Composed panels ────────────────────────────────────────────────────────
    image_panel_t4 = (base_img_dmap_t4 * img_overlay_dmap_t4).opts(
        hv.opts.Overlay(title="Image"),
    )
    phasor_panel_t4 = (ref_arc_t4 * phasor_bg_t4 * ph_overlay_dmap_t4).opts(
        hv.opts.Overlay(
            frame_width=_IMG_WW, frame_height=_IMG_WW,
            xlabel="G", ylabel="S",
            show_grid=True, padding=0.1,
            title="Phasor",
        ),
    )

    return pn.Column(
        combined_warning,
        info_md,
        pn.Row(
            wl_slider_t4,
            pn.Spacer(width=30),
            pn.Row(mode_toggle_t4, reset_btn_t4, sizing_mode="fixed"),
            sizing_mode="fixed",
        ),
        pn.Row(image_panel_t4, phasor_panel_t4),
        spectrum_t4,
    )


# ── Sidebar layout ────────────────────────────────────────────────────────────

global_card = pn.Card(
    start_lambda_input,
    step_lambda_input,
    end_lambda_input,
    harmonic_input,
    show_individual,
    title="Global Parameters",
    collapsed=False,
)

tab1_card = pn.Card(
    t1_mean,
    t1_std,
    title="1) Single Gaussian",
    collapsed=False,
)

tab2_card = pn.Card(
    t2_n_samples,
    t2_mean_base,
    t2_mean_spread,
    t2_std_base,
    t2_std_spread,
    pn.layout.Divider(),
    t2_add_noise,
    t2_snr,
    title="2) Multi Gaussian + Noise",
    collapsed=True,
)

tab3_card = pn.Card(
    t3_file_input,
    title="3) Fluorescence Spectra (CSV)",
    collapsed=True,
)

tab4_card = pn.Card(
    t4_file_input,
    pn.pane.Markdown(
        "**Start λ** and **End λ** from Global Parameters define the wavelength range.  \n"
        "**Step λ** is ignored — it is auto-computed as linspace over the TIFF frame count."
    ),
    title="4) TIFF Image Stack",
    collapsed=True,
)

sidebar = pn.Column(
    global_card,
    tab1_card,
    tab2_card,
    tab3_card,
    tab4_card,
    margin=(10, 10),
)

# ── Main layout ───────────────────────────────────────────────────────────────

header = pn.Column(
    pn.pane.Markdown("# Spectral Phasors Analysis"),
    pn.pane.Markdown(
        "Select a tab to explore different modalities. "
        "Global wavelength range and harmonic apply to all tabs."
    ),
)

main_tabs = pn.Tabs(
    ("1) Single Gaussian", tab1_view),
    ("2) Multi Gaussian + Noise", tab2_view),
    ("3) Fluorescence Spectra (CSV)", tab3_view),
    ("4) TIFF Image Stack", tab4_view),
    dynamic=True,
)

main_area = pn.Column(header, main_tabs, margin=(10, 20))

# ── Sync tab ↔ sidebar card collapse ──────────────────────────────────────────

_tab_cards = {0: tab1_card, 1: tab2_card, 2: tab3_card, 3: tab4_card}


def _sync_cards(event):
    """Collapse every tab-specific card except the one for the active tab.
    Also disable Step λ when the TIFF tab (index 3) is active, since wavelengths
    are derived automatically from linspace(start, end, n_frames).
    """
    for idx, card in _tab_cards.items():
        card.collapsed = idx != event.new
    step_lambda_input.disabled = (event.new == 3)


main_tabs.param.watch(_sync_cards, "active")

# ── Auto-enable show_individual on CSV upload ─────────────────────────────────


def _on_csv_upload(event):
    if event.new is not None:
        show_individual.value = True
        main_tabs.active = 2
        pn.state.location.reload = False
        pn.state.location.reload = True


t3_file_input.param.watch(_on_csv_upload, "value")


def _on_tiff_upload(event):
    if event.new is not None:
        main_tabs.active = 3
        pn.state.location.reload = False
        pn.state.location.reload = True


t4_file_input.param.watch(_on_tiff_upload, "value")

template = pn.template.BootstrapTemplate(
    site="Spectral Analysis",
    title="Phasors App",
    sidebar=[sidebar],
    main=[main_area],
)

template.servable()

if __name__ == "__main__":
    pn.serve(template, show=True)
