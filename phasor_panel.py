import io

import panel as pn
import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv
from holoviews import streams
from itertools import cycle
import colorcet
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

sidebar = pn.Column(
    global_card,
    tab1_card,
    tab2_card,
    tab3_card,
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
    dynamic=True,
)

main_area = pn.Column(header, main_tabs, margin=(10, 20))

# ── Sync tab ↔ sidebar card collapse ──────────────────────────────────────────

_tab_cards = {0: tab1_card, 1: tab2_card, 2: tab3_card}


def _sync_cards(event):
    """Collapse every tab-specific card except the one for the active tab."""
    for idx, card in _tab_cards.items():
        card.collapsed = idx != event.new


main_tabs.param.watch(_sync_cards, "active")

# ── Auto-enable show_individual on CSV upload ─────────────────────────────────


def _on_csv_upload(event):
    if event.new is not None:
        show_individual.value = True
        main_tabs.active = 2
        pn.state.location.reload = False
        pn.state.location.reload = True


t3_file_input.param.watch(_on_csv_upload, "value")

template = pn.template.BootstrapTemplate(
    site="Spectral Analysis",
    title="Phasors App",
    sidebar=[sidebar],
    main=[main_area],
)

template.servable()

if __name__ == "__main__":
    pn.serve(template, show=True)
