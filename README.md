# Spectral Phasors Analysis

A Python application for spectral phasor analysis built with [Panel](https://panel.holoviz.org/) and [HoloViews](https://holoviews.org/). This tool allows you to explore spectral phasors through simulations (single/multi‑Gaussian) and analyze real experimental data via CSV spectra or TIFF hyperspectral stacks.

## Features

- **Simulations**: Interactive single and multi‑component Gaussian simulations with optional noise.
- **Spectra Analysis**: Import CSV files containing spectral data.
- **Image Analysis**: Import TIFF hyperspectral stacks for pixel‑wise phasor analysis with bidirectional selection (Image ↔ Phasor).

## Prerequisites

The project uses [`uv`](https://github.com/astral-sh/uv) for fast Python package management. Ensure it is installed on your system.

## Installation

```bash
# Clone the repository
git clone https://github.com/DBP008/spectral-phasors-analysis.git
cd spectral-phasors-analysis

# Install dependencies and run the app
uv sync
```

## Running the Application

```bash
uv run phasor_panel.py
```

The application will launch automatically in your default browser (usually at `http://localhost:5900`).

## Usage Overview

- **Global Parameters**: Set the wavelength range and harmonic number.
- **Tabs**:
  - **Tab 1** – Single Gaussian simulation.
  - **Tab 2** – Multi‑Gaussian simulation with optional noise.
  - **Tab 3** – Upload a CSV file to visualise experimental spectra.
  - **Tab 4** – Upload a TIFF stack for image‑based phasor analysis.
  - **Tab 5** – Two‑component phasor deconvolution.
  - **Tab 6** – Three‑component phasor deconvolution.
- **Interactive Plots**: Click or box‑select points on the phasor plot to view corresponding spectra, and vice‑versa.

## Sample Data

Sample datasets are provided in the `data/` folder to help you get started:

- `Denaturation.csv`: Example fluorescence spectra data acquired with a **Cary Eclipse Fluorescence Spectrometer**.
- `.tif` files: Example hyperspectral stacks (lambda‑xy scans) acquired with a **Leica confocal microscope**.

## License

This project is licensed under the CC-BY4.0 License.
