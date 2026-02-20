import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    import pandas as pd
    import scipy.stats as stats
    import holoviews as hv
    import hvplot.xarray
    import hvplot.pandas
    import panel as pn

    hv.extension("bokeh")
    return hv, mo, np, pn, xr


@app.cell
def _(np, xr):
    def calculate_phasor_transform(ds: xr.Dataset):
        fft_values = xr.apply_ufunc(
            np.fft.fft,
            ds.intensity,
            input_core_dims=[["wavelength"]],
            output_core_dims=[["harmonic_bin"]],
        )

        # 2. Get the DC component (bin 0) for normalization
        dc_component = fft_values.isel(harmonic_bin=0).real

        # 3. Get the target harmonic index from attributes (default to 1)
        h = ds.attrs.get("harmonic")

        harmonic_data = fft_values.isel(harmonic_bin=h)

        # 5. Assign directly back to the original dataset
        ds["G"] = harmonic_data.real / dc_component
        ds["S"] = harmonic_data.imag / dc_component

    return (calculate_phasor_transform,)


@app.cell
def _(np, xr):
    def add_spectral_reference(ds: xr.Dataset, n_ref_points=None):
        harmonic = int(ds.attrs.get("harmonic"))

        w_min = ds["wavelength"].min().item()
        w_max = ds["wavelength"].max().item()
        spectral_width = w_max - w_min

        if n_ref_points is not None:
            w_ref = np.linspace(w_min, w_max, int(n_ref_points))
        else:
            w_ref = ds["wavelength"].values
        ds.coords["wavelength_ref"] = w_ref

        N = len(w_ref)

        # The FFT of a delta at index 'n' for a specific harmonic 'k' is:
        # exp(-2j * pi * k * n / N)

        indices = np.arange(N)

        phase = -2 * np.pi * harmonic * indices / N

        ds["G_ref"] = (("wavelength_ref"), np.cos(phase))
        ds["S_ref"] = (("wavelength_ref"), np.sin(phase))

        # delta_intensities = np.eye(len(w_ref))

        # fft_results = np.fft.fft(delta_intensities)

        # g_ref = np.real(fft_results[harmonic, :])
        # s_ref = np.imag(fft_results[harmonic, :])

        # ds["G_ref"] = (("wavelength_ref"), g_ref)
        # ds["S_ref"] = (("wavelength_ref"), s_ref)
    return (add_spectral_reference,)


@app.function
def validate_setup_params(values):
    """
    Validates the wavelength setup parameters.
    Returns: str | None: Error message if invalid, None if valid.
    """
    if not values:
        return "Form has no values."

    try:
        start = float(values["start_lambda"])
        step = float(values["step_lambda"])
        end = float(values["end_lambda"])
        harmonic = float(values["harmonic_N"])
    except ValueError:
        return "All parameters must be valid numbers."

    if step <= 0:
        return "Step Lambda must be a positive number."

    if start >= end:
        return "Start Lambda must be less than End Lambda."

    if harmonic < 1 or not harmonic.is_integer():
        return "Harmonic N must be a positive integer (1, 2, 3...)."

    return None


@app.cell
def _(mo):
    mo.md(r"""
    # Spectral Phasors Analysis

    This interactive notebook demonstrates the concept of **Spectral Phasors**.

    Spectral phasors transform spectral data into a 2D coordinate system using the Fast Fourier Transform (FFT).
    """)
    return


@app.cell
def _(mo):
    parameters_form = (
        mo.md("""
            ## Please insert the setup parameters below:

            | Parameter | Value |
            | ---: | :--- |
            | **Start Lambda:** | {start_lambda} $nm$ |
            | **Step Lambda:** | {step_lambda} $nm$ |
            | **End Lambda:** | {end_lambda} $nm$ |
            | **Harmonic N:** | {harmonic_N} |
        """)
        .batch(
            start_lambda=mo.ui.text(value="400"),
            step_lambda=mo.ui.text(value="0.5"),  # Fixed typo in 'lambda'
            end_lambda=mo.ui.text(value="700"),
            harmonic_N=mo.ui.text(value="1"),
        )
        .form(show_clear_button=True)
    )

    parameters_form.center()
    return (parameters_form,)


@app.cell
def _(
    add_spectral_reference,
    calculate_phasor_transform,
    mo,
    np,
    parameters_form,
    xr,
):
    mo.stop(not parameters_form.value)

    err = validate_setup_params(parameters_form.value)
    if isinstance(err, str):
        mo.output.append(mo.callout(str(err), kind="danger"))
    else:
        mo.output.append(mo.callout("Ok!", kind="success"))

        start = float(parameters_form.value["start_lambda"])
        step = float(parameters_form.value["step_lambda"])
        end = float(parameters_form.value["end_lambda"])
        harmonic = float(parameters_form.value["harmonic_N"])

        _wavelength_coords = np.arange(start, end + step / 2, step)

        wavelengths = np.linspace(400, 700, 256)
        sample_indices = np.arange(200)

        spectra_xr_ds = xr.Dataset(
            data_vars={
                # Intensity is 2D: (sample x wavelength)
                "intensity": (
                    ["sample", "wavelength"],
                    # np.full((len(sample_indices), len(wavelengths)), np.nan),
                    np.random.rand(len(sample_indices), len(wavelengths)),
                ),
                # Phasors are 1D: (sample)
                "G": (["sample"], np.full(len(sample_indices), np.nan)),
                "S": (["sample"], np.full(len(sample_indices), np.nan)),
            },
            coords={
                "sample": sample_indices,
                "wavelength": wavelengths,
            },
        )

        spectra_xr_ds.attrs["harmonic"] = 1
        # Add units immediately so you don't forget
        spectra_xr_ds.wavelength.attrs = {"units": "nm"}

        spectra_xr_ds.G.attrs = {"long_name": "Phasor G (real)"}
        spectra_xr_ds.S.attrs = {"long_name": "Phasor S (imaginary)"}

        calculate_phasor_transform(spectra_xr_ds)
        add_spectral_reference(spectra_xr_ds, 1e5)
    return (spectra_xr_ds,)


@app.cell
def _(spectra_xr_ds):
    print(spectra_xr_ds)
    return


@app.cell
def _(df_lines):
    print(df_lines.head())
    return


@app.cell
def _(df_lines, hv, spectra_xr_ds):
    from holoviews.selection import link_selections

    df = spectra_xr_ds[["G", "S"]].to_dataframe().reset_index()
    _hv_ds = hv.Dataset(df)
    _scatter = hv.Points(_hv_ds, kdims=["G", "S"], vdims=["sample"])
    _scatter.opts(
        color="sample",
        cmap="Category20",
        size=8,
        width=500,
        height=500,
        tools=["hover"],
        colorbar=True,
        title="Phasor Plot",
    )
    # _scatter

    _hv_ds_lines = hv.Dataset(df_lines)

    _lines = _hv_ds_lines.to(
        hv.Curve, "wavelength", "intensity", "sample"
    ).overlay()

    _linker = link_selections.instance()

    _scatter + _lines
    return (link_selections,)


@app.cell
def _(spectra_xr_ds):
    df_scatter = spectra_xr_ds[["G", "S"]].to_dataframe().reset_index()

    df_lines = spectra_xr_ds["intensity"].to_dataframe().reset_index()
    return df_lines, df_scatter


@app.cell
def _(df_scatter):
    df_scatter
    return


@app.cell
def _(df_scatter, hv, pn, spectra_xr_ds):
    from holoviews import streams

    # 1. Initialize Panel
    pn.extension()

    # df_scatter = spectra_xr_ds[["G", "S"]].to_dataframe().reset_index()

    # 2. Scatter Plot and Selection
    _scatter = hv.Points(df_scatter, kdims=["G", "S"], vdims=["sample"]).opts(
        color="sample",
        cmap="Category20",
        size=8,
        tools=["box_select", "lasso_select", "tap", "hover"],
        width=450,
        height=450,
        title="1. Select Points",
    )

    selection_test = streams.Selection1D(source=_scatter)
    _scatter

    _line = (
        spectra_xr_ds.intensity.sel(sample=selection_test.index)
        .mean("sample", keep_attrs=True)
        .hvplot.line(x="wavelength", y="intensity")
    )

    _scatter + _line
    return selection_test, streams


@app.cell
def _(hv, np, pn):
    hv.extension("bokeh")
    pn.extension()

    # 1. Create your data/element
    points = hv.Points(np.random.rand(100, 2))

    # 2. Define the Selection1D stream linked to the points
    selection = hv.streams.Selection1D(source=points)


    # 3. Define a callback that uses the 'index' variable
    def callback(index):
        # 'index' is a list of integer indices (e.g., [1, 5, 22])
        if not index:
            return hv.Text(0.5, 0.5, "No selection")

        # Access the actual data using .iloc
        selected_data = points.iloc[index]
        mean_y = selected_data.dimension_values(1).mean()

        return hv.Text(
            0.5, 0.5, f"Selected: {len(index)} points\nMean Y: {mean_y:.2f}"
        )


    # 4. Link everything in a DynamicMap
    dmap = hv.DynamicMap(callback, streams=[selection])

    # Display both (ensure tools include selection tools)
    points.opts(tools=["tap", "box_select", "lasso_select"], size=10) + dmap

    _layout = pn.panel(points + dmap)

    # Make it a serveable app
    _layout
    return (dmap,)


@app.cell
def _():
    return


@app.cell
def _(df_scatter, hv, pn, spectra_xr_ds, streams):
    # 1. Initialize Panel
    pn.extension()

    # 2. Define the Scatter Plot
    _scatter = hv.Points(df_scatter, kdims=["G", "S"], vdims=["sample"]).opts(
        color="sample",
        cmap="Category20",
        size=8,
        tools=["box_select", "lasso_select", "tap", "hover"],
        width=450,
        height=450,
        title="1. Select Points",
    )

    # 3. Create the Selection Stream linked to the scatter plot
    select_1 = streams.Selection1D(source=_scatter)


    @pn.depends(select_1.param.index)
    def reactive_line(index):
        print(index)
        return (
            spectra_xr_ds.intensity.isel(sample=index).mean("sample").hvplot.line()
        )


    pn.Row(_scatter, reactive_line)
    return (select_1,)


@app.cell
def _(select_1):
    select_1.index
    return


@app.cell
def _(spectra_xr_ds):
    spectra_xr_ds.intensity.sel(sample=[]).mean("sample", keep_attrs=True)
    return


@app.cell
def _(spectra_xr_ds):
    spectra_xr_ds.intensity.sel(sample=[]).mean(
        "sample", keep_attrs=True
    ).hvplot.line(x="wavelength", y="intensity")
    return


@app.cell
def _(df_lines, selection_test):
    selected_lines = df_lines[
        df_lines["sample"].isin(selection_test.index)
    ].reset_index()
    selected_lines
    return (selected_lines,)


@app.cell
def _(hv, selected_lines):
    _plot_lines = (
        hv.Dataset(selected_lines)
        .to(hv.Curve, "wavelength", "intensity", "sample")
        .overlay()
    )
    _plot_lines
    return


@app.cell
def _(df_lines, df_scatter, hv, link_selections, pn):
    hv.extension("bokeh")
    pn.extension()

    # 2. Define the Datasets
    # Important: We tell HoloViews that 'sample' is a Dimension (kdims) in both
    ds_scatter = hv.Dataset(df_scatter, kdims=["G", "sample"], vdims=["S"])
    ds_lines = hv.Dataset(
        df_lines, kdims=["wavelength", "sample"], vdims=["intensity"]
    )

    # 3. Create the plots
    # Note: We use .overlay('sample') for the lines so HoloViews knows
    # each 'sample' is a distinct object to be filtered.
    scatter_plot = hv.Points(ds_scatter).opts(
        color="sample",
        cmap="Category20",
        size=10,
        width=400,
        height=400,
        tools=["box_select", "lasso_select", "tap"],
    )

    line_plot = (
        ds_lines.to(hv.Curve, "wavelength", "intensity")
        .overlay("sample")
        .opts(width=600, height=400, show_legend=False)
    )

    # 4. Create the Linker object
    # This object manages the selection state between the two plots
    linker = link_selections.instance()

    # 5. Apply the linker to the plots and display
    # This automatically handles the "index to sample" logic for you
    layout = linker(scatter_plot) + linker(line_plot)

    layout
    return


@app.cell
def _(dmap):
    dmap.event()
    return


@app.cell
def _(df_lines):
    df_lines.hvplot.line(
        x="wavelength",
        y="intensity",
        by="sample",
        line_width=1,
        alpha=0.1,
        legend=False,
        width=600,
        height=400,
        title="Spectra (Selection filtered)",
    )
    return


@app.cell
def _():
    # from holoviews import selection

    # hv.extension("bokeh")

    # # 1. Prepare the DataFrames
    # # Dataframe for the Scatter (1 row per sample)


    # # 2. Create the Selection Link instance
    # # This object synchronizes selections across any plot it is applied to
    # ls = selection.link_selections.instance()

    # # 3. Define the Scatter Plot
    # # Use 'sample' as a vdim so the selection tool knows how to map it
    # _scatter = df_scatter.hvplot.scatter(
    #     x="G",
    #     y="S",
    #     c="sample",
    #     cmap="Category20",
    #     size=50,
    #     width=400,
    #     height=400,
    #     title="Phasor Plot (Select here!)",
    # )

    # # 4. Define the Line Plot
    # # We 'by' sample so that each sample is an individual line
    # _lines = df_lines.hvplot.line(
    #     x="wavelength",
    #     y="intensity",
    #     by="sample",
    #     line_width=1,
    #     alpha=0.4,  # Make lines slightly transparent so overlaps are visible
    #     legend=False,
    #     width=600,
    #     height=400,
    #     title="Spectra (Selection filtered)",
    # )

    # # 5. Apply the Link and Display
    # # We use the '+' operator to put them side-by-side and wrap in 'ls'
    # # layout = ls(_scatter + _lines)

    # # layout.opts(shared_axes=False)

    # # layout
    return


@app.cell
def _(spectra_xr_ds):
    spectra_xr_ds.intensity.hvplot.line(
        x="wavelength",
        by="sample",
        cmap="Category20",
        legend=False,  # 200 items in a legend can be messy
        width=700,
        height=400,
    )
    return


@app.cell
def _(spectra_xr_ds):
    spectra_xr_ds["intensity"].to_dataframe().reset_index()
    return


@app.cell
def _(hv, pn):
    from holoviews import opts


    # Initialize extensions
    hv.extension("bokeh")


    def create_linked_dashboard(spectra_xr_ds, sample_cmap="Set1"):
        # 1. Prepare Data
        # ---------------------------------------------------------
        # Phasor Data
        df_samples = spectra_xr_ds[["G", "S"]].to_dataframe().reset_index()

        # Reference Circle Data
        df_ref = (
            spectra_xr_ds[["G_ref", "S_ref", "wavelength_ref"]]
            .to_dataframe()
            .reset_index()
        )

        # Spectra Data
        # Melting the intensity array so it's tidy: sample, wavelength, intensity
        df_spectra = spectra_xr_ds["intensity"].to_dataframe().reset_index()

        # 2. Define Element Elements
        # ---------------------------------------------------------

        # Phasor Scatter
        # We include 'sample' in vdims so the linker can use it to sync
        sample_scatter = hv.Scatter(
            df_samples, kdims=["G"], vdims=["S", "sample"]
        ).opts(
            size=10,
            color="sample",
            cmap=sample_cmap,
            tools=["hover", "box_select", "lasso_select", "tap"],
            nonselection_alpha=0.1,
            title="Phasor Plot",
        )

        # Reference Background (Static)
        ref_chart = hv.Scatter(
            df_ref, kdims=["G_ref"], vdims=["S_ref", "wavelength_ref"]
        ).opts(
            color="wavelength_ref",
            cmap="rainbow4",
            size=2,
            alpha=0.3,
            colorbar=False,
        )

        # Spectra Curves
        # We create a Dataset first to make the mapping clear
        ds_spectra = hv.Dataset(
            df_spectra, kdims=["wavelength", "sample"], vdims=["intensity"]
        )

        # .to() creates a Curve for every 'sample', .overlay() puts them on one plot
        spectra_overlay = ds_spectra.to(
            hv.Curve, "wavelength", "intensity"
        ).overlay("sample")

        # Apply Options to Spectra
        # Note: We remove 'cmap' from Curve and use it within the NdOverlay context
        spectra_overlay.opts(
            opts.Curve(
                color="sample",
                # cmap=sample_cmap,
                line_width=1.5,
                alpha=0.6,
                tools=["hover", "box_select", "tap"],
                nonselection_alpha=0.05,  # This creates the "masking" effect
                nonselection_color="grey",
            ),
            opts.NdOverlay(
                width=600, height=400, show_legend=False, title="Intensity Spectra"
            ),
        )

        # 3. Create Selection Linker
        # ---------------------------------------------------------
        # This tool links the two plots based on the 'sample' dimension
        linker = hv.selection.link_selections.instance()

        # Wrap our interactive elements in the linker
        # Note: We overlay the linked scatter on the static reference
        linked_phasor = ref_chart * linker(sample_scatter)
        linked_spectra = linker(spectra_overlay)

        # 4. Final Layout
        # ---------------------------------------------------------
        dashboard = pn.Row(
            linked_phasor.opts(width=450, height=450, padding=0.1), linked_spectra
        )

        return dashboard


    # To display in a marimo or jupyter cell:
    # create_linked_dashboard(spectra_xr_ds)
    return


@app.cell
def _():
    # create_linked_dashboard(spectra_xr_ds)
    return


@app.cell
def _():
    # from holoviews.selection import link_selections


    # def create_linked_phasor_app(spectra_xr_ds, sample_cmap="Set1"):
    #     hv.extension("bokeh")

    #     # 1. Detect Dimensions
    #     # We assume G has 1 dimension (the samples)
    #     dim_name = spectra_xr_ds.G.dims[0]

    #     # 2. Prepare Dataframes
    #     # Ensure all coordinates become columns
    #     df_samples = spectra_xr_ds[["G", "S"]].to_dataframe().reset_index()
    #     df_spectra = spectra_xr_ds["intensity"].to_dataframe().reset_index()

    #     # Determine which column to use for coloring
    #     # If 'sample' was a coordinate in Xarray, it is now a column.
    #     # Otherwise, we use the index dimension.
    #     color_column = 'sample' if 'sample' in df_samples.columns else dim_name

    #     # 3. Create Datasets
    #     ds_samples = hv.Dataset(df_samples)
    #     ds_spectra = hv.Dataset(df_spectra)

    #     # 4. Prepare Reference Background (The Circle/Arc)
    #     df_ref = spectra_xr_ds[["G_ref", "S_ref", "wavelength_ref"]].to_dataframe().reset_index()
    #     ref_chart = hv.Scatter(df_ref, kdims=["G_ref"], vdims=["S_ref", "wavelength_ref"]).opts(
    #         color="wavelength_ref", cmap="rainbow4", size=2, alpha=0.1, tools=[]
    #     )

    #     # 5. Define Plots
    #     # Phasor Scatter Plot
    #     phasor_scatter = hv.Scatter(ds_samples, kdims=["G"], vdims=["S", color_column, dim_name]).opts(
    #         color=color_column,
    #         cmap=sample_cmap,
    #         size=10,
    #         tools=["hover", "lasso_select", "box_select", "tap"],
    #         nonselection_alpha=0.1,
    #         show_legend=True,
    #         width=400, height=400
    #     )

    #     # Intensity Spectra Plot
    #     # We overlay the curves grouped by the sample dimension
    #     # We use Cycle(sample_cmap) to ensure line colors match the scatter colors
    #     spectra_curves = ds_spectra.to(hv.Curve, 'wavelength', 'intensity').overlay(dim_name).opts(
    #         hv.opts.Curve(color=hv.Cycle(sample_cmap), line_width=1.5, alpha=0.7),
    #         hv.opts.NdOverlay(show_legend=False, title="Intensity Spectra", width=500, height=400)
    #     )

    #     # 6. Apply the Linkage
    #     linker = link_selections.instance()

    #     # Selection on phasor_scatter filters ds_spectra because they share the same 'dim_name' column
    #     phasor_panel = ref_chart * linker(phasor_scatter)
    #     spectra_panel = linker(spectra_curves)

    #     layout = (phasor_panel + spectra_panel).opts(
    #         hv.opts.Layout(shared_axes=False, merge_tools=False)
    #     )

    #     return layout

    # # To display in Marimo/Jupyter:
    # # create_linked_phasor_app(spectra_xr_ds)
    return


@app.cell
def _(create_linked_phasor_app, spectra_xr_ds):
    create_linked_phasor_app(spectra_xr_ds)
    return


@app.cell
def _(hv, spectra_xr_ds):
    _test = (
        spectra_xr_ds[["G_ref", "S_ref", "wavelength_ref"]]
        .to_dataframe()
        .reset_index()
    )

    hv.Dataset(_test, ["G_ref", "S_ref", "wavelength_ref"])
    return


@app.cell
def _(hv):
    def create_phasor_plot(spectra_xr_ds, sample_cmap="Set1"):
        hv.extension("bokeh")

        # 1. Prepare Sample Data
        df_samples = spectra_xr_ds[["G", "S"]].to_dataframe()
        ds_samples = hv.Dataset(df_samples)

        # 2. Prepare Reference Data
        df_ref = (
            spectra_xr_ds[["G_ref", "S_ref", "wavelength_ref"]]
            .to_dataframe()
            .reset_index()
        )
        ds_ref = hv.Dataset(df_ref)

        # 3. Create Reference Chart (Background)
        ref_chart = hv.Scatter(
            ds_ref, kdims=["G_ref"], vdims=["S_ref", "wavelength_ref"]
        ).opts(
            color="wavelength_ref",
            cmap="rainbow4",
            size=4,
            colorbar=False,
            tools=["hover"],
            clabel="Wavelength (nm)",
        )

        # 4. Create Sample Chart (Foreground)
        sample_chart = hv.Scatter(
            ds_samples, kdims=["G"], vdims=["S", "sample"]
        ).opts(
            color="sample",
            cmap=sample_cmap,
            marker="circle",
            size=10,
            tools=["hover", "box_select", "lasso_select"],
            show_legend=True,
            legend_position="right",
        )

        # 5. Combine and apply global options
        phasor_plot = (ref_chart * sample_chart).opts(
            hv.opts.Scatter(
                frame_width=400,
                frame_height=400,
                padding=0.1,
                xlabel="G",
                ylabel="S",
                show_grid=True,
                title="Phasor Plot",
            )
        )

        return phasor_plot


    # Usage:
    # plot = create_phasor_plot(spectra_xr_ds)
    # plot
    return (create_phasor_plot,)


@app.cell
def _(create_phasor_plot, spectra_xr_ds):
    tmp = create_phasor_plot(spectra_xr_ds)
    tmp
    return (tmp,)


@app.cell
def _(tmp):
    tmp.children
    return


@app.cell
def _(tmp):
    tmp
    return


@app.cell
def _(add_spectral_reference, calculate_phasor_transform, np, xr):
    def create_mockup_phasor_dataset(width=100, height=100, n_wl=256):
        # 1. Define coordinates
        x = np.arange(width)
        y = np.arange(height)
        wavelength = np.linspace(400, 700, n_wl)

        # 2. Create simulated spectral data (Gaussians)
        # We'll make the peak position vary across the image (x-direction)
        # and the intensity vary across the image (y-direction)
        X, Y, WL = np.meshgrid(x, y, wavelength, indexing="ij")

        # Center of Gaussian shifts from 500nm to 600nm across the X axis
        center = 500 + (X / width) * 100
        sigma = 30

        # Intensity decays along the Y axis
        amplitude = 1000 * np.exp(-Y / height)

        # Generate the 3D intensity array (x, y, wavelength)
        intensity_data = amplitude * np.exp(-((WL - center) ** 2) / (2 * sigma**2))

        # Add a bit of random noise
        intensity_data += np.random.normal(0, 10, intensity_data.shape)
        intensity_data = np.clip(
            intensity_data, 0, None
        )  # Remove negative values from noise

        # 3. Assemble the Dataset
        ds = xr.Dataset(
            data_vars={
                "intensity": (["x", "y", "wavelength"], intensity_data),
                # Placeholder variables for G and S (optional, function will overwrite)
                "G": (["x", "y"], np.full((width, height), np.nan)),
                "S": (["x", "y"], np.full((width, height), np.nan)),
            },
            coords={
                "x": x,
                "y": y,
                "wavelength": wavelength,
            },
            attrs={"harmonic": 1, "units": "nm"},
        )

        return ds


    # --- Execution ---
    # 1. Generate the mockup
    ds_image = create_mockup_phasor_dataset()
    calculate_phasor_transform(ds_image)
    add_spectral_reference(ds_image)

    ds_image
    return (ds_image,)


@app.cell
def _(ds_image):
    print(ds_image)
    return


@app.cell
def _():
    # from holoviews import opts
    # from holoviews.operation.datashader import rasterize
    # import datashader as ds


    # def create_phasor_dashboard(xr_ds):
    #     hv.extension("bokeh")

    #     # --- 1. Reference Plot (Rasterized for 1M points) ---
    #     df_ref = (
    #         xr_ds[["G_ref", "S_ref", "wavelength_ref"]]
    #         .to_dataframe()
    #         .reset_index()
    #     )
    #     # Use Points instead of Scatter for 2D coordinates
    #     ref_points = hv.Points(
    #         df_ref, kdims=["G_ref", "S_ref"], vdims=["wavelength_ref"]
    #     )

    #     ref_chart = rasterize(
    #         ref_points, aggregator=ds.mean("wavelength_ref")
    #     ).opts(
    #         cmap="Spectral_r",
    #         colorbar=False,
    #         title="Phasor Reference",
    #         xlabel="G",
    #         ylabel="S",
    #         width=400,
    #         height=400,
    #         tools=["hover"],
    #     )

    #     # --- 2. Mode Detection ---
    #     is_image_mode = "x" in xr_ds.coords and "y" in xr_ds.coords
    #     link = hv.link_selections.instance()

    #     if is_image_mode:
    #         # ================= IMAGE MODE =================
    #         ds_intensity = hv.Dataset(xr_ds["intensity"])
    #         image_scrubber = ds_intensity.to(hv.Image, kdims=["x", "y"]).opts(
    #             cmap="viridis",
    #             tools=["box_select"],
    #             active_tools=["box_select"],
    #             title="Intensity (Wavelength Slider)",
    #             frame_width=400,
    #             frame_height=400,
    #         )

    #         df_phasor = xr_ds[["G", "S"]].to_dataframe().reset_index().dropna()
    #         # Use hv.Points for 2D G/S coordinates
    #         phasor_points = hv.Points(
    #             df_phasor, kdims=["G", "S"], vdims=["x", "y"]
    #         ).opts(
    #             color="gray",
    #             alpha=0.3,
    #             size=3,
    #             tools=["box_select", "lasso_select"],
    #             nonselection_alpha=0.05,
    #             selection_color="red",
    #         )

    #         linked_layout = link(image_scrubber + phasor_points)

    #         def get_mean_spectrum(selection_expr):
    #             subset = (
    #                 link.filter(xr_ds, selection_expr=selection_expr)
    #                 if selection_expr
    #                 else xr_ds
    #             )
    #             mean_vals = subset["intensity"].mean(dim=["x", "y"])
    #             return hv.Curve(
    #                 mean_vals, kdims=["wavelength"], vdims=["intensity"]
    #             )

    #         spectrum_view = hv.DynamicMap(
    #             get_mean_spectrum,
    #             streams={"selection_expr": link.param.selection_expr},
    #         ).opts(
    #             color="black",
    #             line_width=2,
    #             responsive=True,
    #             height=250,
    #             title="Mean Spectrum of Selection",
    #             framewise=True,
    #         )

    #         phasor_overlay = ref_chart * linked_layout[1]
    #         dashboard = (linked_layout[0] + phasor_overlay).cols(2) + spectrum_view

    #     else:
    #         # ================= SPECTRA (SAMPLE) MODE =================
    #         df_phasor = xr_ds[["G", "S"]].to_dataframe().reset_index().dropna()

    #         # Using Points avoids the "Chart elements should only be supplied a single kdim" warning
    #         phasor_points = hv.Points(
    #             df_phasor, kdims=["G", "S"], vdims=["sample"]
    #         ).opts(
    #             color="sample",
    #             cmap="Category20",
    #             size=7,
    #             tools=["box_select", "tap"],
    #             nonselection_alpha=0.1,
    #         )

    #         ds_spectra = hv.Dataset(
    #             xr_ds, kdims=["sample", "wavelength"], vdims=["intensity"]
    #         )
    #         spectra_overlay = ds_spectra.to(hv.Curve, "wavelength").overlay(
    #             "sample"
    #         )

    #         # Apply opts separately. Note: removed 'cmap' from Curve and used hv.Cycle
    #         spectra_overlay = spectra_overlay.opts(
    #             opts.Curve(
    #                 color=hv.Cycle(
    #                     "Category20"
    #                 ),  # Match the colormap of the points
    #                 line_width=1.5,
    #                 nonselection_alpha=0.05,
    #                 tools=["tap"],
    #             ),
    #             opts.NdOverlay(
    #                 show_legend=False,
    #                 height=350,
    #                 responsive=True,
    #                 title="Sample Spectra (Highlight by selection)",
    #             ),
    #         )

    #         linked_layout = link(phasor_points + spectra_overlay)

    #         phasor_panel = ref_chart * linked_layout[0]
    #         dashboard = phasor_panel + linked_layout[1]

    #     return dashboard.opts(opts.Layout(shared_axes=False, merge_tools=True))


    # create_phasor_dashboard(spectra_xr_ds)
    return


@app.cell
def _():
    # def create_interactive_phasor_dashboard_old(xr_ds):
    #     hv.extension("bokeh")

    #     # 1. Reference Data
    #     df_ref = (
    #         xr_ds[["G_ref", "S_ref", "wavelength_ref"]]
    #         .to_dataframe()
    #         .reset_index()
    #     )
    #     ds_ref = hv.Dataset(df_ref)

    #     ref_chart = hv.Scatter(
    #         ds_ref, kdims=["G_ref"], vdims=["S_ref", "wavelength_ref"]
    #     ).opts(
    #         color="wavelength_ref",
    #         cmap="Spectral_r",
    #         size=4,
    #         colorbar=False,
    #         tools=["hover"],
    #     )

    #     # 2. Sample Points
    #     df_samples = xr_ds[["G", "S"]].to_dataframe().reset_index().dropna()
    #     phasor_points = hv.Points(
    #         df_samples, kdims=["G", "S"], vdims=["x", "y"]
    #     ).opts(
    #         color="gray",
    #         alpha=0.3,
    #         size=5,
    #         tools=["box_select", "lasso_select"],
    #         nonselection_alpha=0.05,
    #         selection_color="red",
    #     )

    #     # 3. Spatial Image
    #     total_intensity = xr_ds["intensity"].sum(dim="wavelength")
    #     spatial_image = hv.Image(
    #         total_intensity, kdims=["x", "y"], vdims=["intensity"]
    #     ).opts(
    #         cmap="Viridis",
    #         tools=["box_select"],
    #         title="Spatial Intensity (Sum)",
    #         aspect="equal",
    #     )

    #     # 4. Setup Linking
    #     link = hv.link_selections.instance()
    #     linked_layout = link(spatial_image + phasor_points)

    #     # 5. Dynamic Spectrum Calculation
    #     # We use the 'selection_expr' parameter from the link instance.
    #     # When a selection changes, this parameter updates, triggering the DynamicMap.
    #     def get_mean_spectrum(selection_expr):
    #         # link.filter uses the selection_expr to subset the dataset
    #         selected_data = link.filter(xr_ds, selection_expr=selection_expr)

    #         # Calculate mean spectrum over spatial dimensions
    #         mean_spec = selected_data["intensity"].mean(dim=["x", "y"])
    #         return hv.Curve(mean_spec, kdims=["wavelength"], vdims=["intensity"])

    #     # Create DynamicMap linked to the selection state
    #     dynamic_spectrum = hv.DynamicMap(
    #         get_mean_spectrum,
    #         streams={"selection_expr": link.param.selection_expr},
    #     ).opts(
    #         color="black",
    #         line_width=2,
    #         responsive=True,
    #         height=300,
    #         title="Mean Spectrum of Selection",
    #         framewise=True,
    #     )

    #     # 6. Final Layout
    #     img_element = linked_layout[0]
    #     points_element = linked_layout[1]

    #     phasor_overlay = (ref_chart * points_element).opts(
    #         title="Phasor Plot", xlabel="G", ylabel="S"
    #     )

    #     dashboard = (img_element + phasor_overlay).cols(2) + dynamic_spectrum

    #     return dashboard.opts(
    #         hv.opts.Scatter(frame_width=400, frame_height=400, show_grid=True),
    #         hv.opts.Image(frame_width=400, frame_height=400),
    #         hv.opts.Layout(shared_axes=False, merge_tools=True),
    #     )
    return


@app.cell
def _():
    # create_interactive_phasor_dashboard(ds_image)
    return


if __name__ == "__main__":
    app.run()
