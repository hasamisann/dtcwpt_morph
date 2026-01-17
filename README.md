# DT-CWPT Morph

**DT-CWPT Morph** is an experimental audio plugin that performs real-time structural audio morphing using the **Dual-Tree Complex Wavelet Packet Transform (DT-CWPT)**.

Unlike traditional FFT-based spectral processors, DT-CWPT provides shift-invariant and directionally selective analysis, allowing for high-fidelity transient preservation and unique textural manipulation capabilities.

## Features

### Dual-Tree Complex Wavelet Processing
- Utilizes the Kingsbury Q-Shift filters for near-perfect reconstruction.
- Shift-invariant properties minimize artifacts during dynamic processing.
- Separation of Magnitude and Phase components for independent morphing.
- Customizable decomposition tree.

### Parameters
- Magnitude: Controls the amplitude morphing factor.
- Phase: Controls the phase morphing factor.
- Threshold: Attenuates phase morphing for signals below this level to prevent noise artifacts.
- Bypass Low/High: Selectively bypass processing for the lowest or highest frequency bands.

## Building

This project is built with Rust and the nih-plug framework.

### Prerequisites
- [Rust Toolchain](https://rustup.rs/) (Stable)
- [CMake](https://cmake.org/) (Required for some dependencies)

### Cloning
```bash
git clone https://github.com/hasamisann/dtcwpt_morph.git
cd dtcwpt_morph
```
