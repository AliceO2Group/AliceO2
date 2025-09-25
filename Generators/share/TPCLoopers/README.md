# TPC Loopers Generator - Parameter Files

This directory contains parameter files used by the TPC Loopers event generator in ALICE O2.

## Overview

The TPC Loopers generator uses pre-trained ONNX models to generate realistic looper particles based on machine learning models trained on full GEANT4 slow neutron transport simulations. The parameter files in this directory provide:
- Example statistical distribution parameters for sampling the number of loopers per event
- **Mandatory** scaling parameters for transforming the ONNX model outputs to physical values

## Files Description

### Statistical Sampling Parameters

The files provided in the folder are examples based on the training dataset.

#### `gaussian_params.csv`
Parameters for Gaussian distribution used to sample the number of Compton electrons per event.

**Format:** Four values (one per line)
1. Mean (μ)
2. Standard deviation (σ)
3. Minimum value
4. Maximum value

#### `poisson_params.csv`
Parameters for Poisson distribution used to sample the number of electron-positron pairs per event.

**Format:** Three values (one per line)
1. Lambda (λ) parameter
2. Minimum value
3. Maximum value

### Scaler Parameters

These JSON files contain the parameters for inverse transformation of the ONNX models output. They should be kept as they are
unless a new version of the models is released.

#### `ScalerComptonParams.json`
Scaler parameters for Compton electron generation model.

**Structure:**
```json
{
  "normal": {
    "min": [array of 5 min values for min-max normalization],
    "max": [array of 5 max values for min-max normalization]
  },
  "outlier": {
    "center": [array of 2 center values for robust scaling],
    "scale": [array of 2 scale values for robust scaling]
  }
}
```

- **normal**: Min-max normalization parameters for standard features (`Px`, `Py`, `Pz`, `VertexCoordinatesX`, `VertexCoordinatesY`)
- **outlier**: Robust scaler parameters (center and scale) for outlier features (`VertexCoordinatesZ`,`time`)

#### `ScalerPairParams.json`
Scaler parameters for electron-positron pair generation model.

**Structure:**
```json
{
  "normal": {
    "min": [array of 8 min values for min-max normalization],
    "max": [array of 8 max values for min-max normalization]
  },
  "outlier": {
    "center": [array of 2 center values for robust scaling],
    "scale": [array of 2 scale values for robust scaling]
  }
}
```

- **normal**: Min-max normalization parameters for standard features (`Px_e`, `Py_e`, `Pz_e`,`Px_p`, `Py_p`, `Pz_p`, `VertexCoordinatesX`, `VertexCoordinatesY`)
- **outlier**: Robust scaler parameters (center and scale) for outlier features (`VertexCoordinatesZ`,`time`)
---
*Author: M. Giacalone - September 2025*
