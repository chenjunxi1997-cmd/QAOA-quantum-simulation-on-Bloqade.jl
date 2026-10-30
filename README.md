# QAOA-quantum-simulation-on-Bloqade.jl
Source code for the paper: Analog quantum approximate optimization algorithm for solving the weighted MaxCut problem on three-dimensional single-atom Rydberg array

This repository contains the source code for the paper:

**"Analog quantum approximate optimization algorithm for solving the weighted MaxCut problem on three-dimensional single-atom Rydberg array"**

*Authors: Chen Junxi*
*Target journal: Quantum*
*Year: 2026 (in preparation)*

---

## Overview

This project implements an analog quantum approximate optimization algorithm (analog-QAOA) to solve the **weighted MaxCut problem** using **three-dimensional single-atom Rydberg arrays**. The code is written in [Julia](https://julialang.org/) and leverages the [Bloqade](https://github.com/QuEraComputing/Bloqade.jl) framework for quantum simulation of Rydberg atom arrays, along with [GenericTensorNetworks](https://github.com/QuEraComputing/GenericTensorNetworks.jl) for exact tensor-network-based solutions to combinatorial optimization problems.

---

## Requirements

- **Julia** >= 1.9 (recommended: 1.10 or later)
- The following Julia packages:

| Package | Purpose |
|---------|---------|
| `GenericTensorNetworks` | Exact solutions via tensor networks |
| `Bloqade` | Rydberg atom array quantum simulation |
| `LinearAlgebra` | Linear algebra operations |
| `Random` | Random number generation |
| `Statistics` | Statistical analysis |
| `Optim` | Classical optimization (for QAOA parameter optimization) |
| `SpecialFunctions` | Special mathematical functions |
| `Printf` | Formatted output |
| `FFTW` | Fast Fourier transforms |
| `Serialization` | Data serialization/saving |
| `Dates` | Timestamp utilities |

> **Note on Bloqade/CondaPkg:** If you encounter CondaPkg-related issues with Bloqade, you can set the environment variable to skip CondaPkg:
> ```bash
> export JULIA_CONDAPKG_BACKEND=Null
> ```

---

## Installation

### 1. Install Julia

Download and install Julia from [https://julialang.org/install/](https://julialang.org/install/).

### 2. Clone this repository

```bash
git clone [https://github.com/yourusername/quantum-maxcut-3d-rydberg.git](https://github.com/chenjunxi1997-cmd/QAOA-quantum-simulation-on-Bloqade.jl.git)
cd quantum-maxcut-3d-rydberg
