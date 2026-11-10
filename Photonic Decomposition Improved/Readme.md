# Photonic Mesh Decomposition: Reck vs. Clements Architectures

## Overview

This project implements and evaluates two fundamental architectures for decomposing arbitrary unitary matrices into cascaded Mach–Zehnder interferometers (MZIs): **Reck and Clements mesh architectures**.

The implementation provides a self-contained Python framework for optical unitary decomposition, reconstruction, and performance analysis. It focuses on replacing numerical optimization with closed-form mathematical solutions, improving computational efficiency, and evaluating the effects of optical insertion loss and phase-setting errors.

Both architectures are implemented from scratch using NumPy, without relying on external photonic neural network decomposition packages.

## Objectives

The main objectives of this project are:

- Implement Reck and Clements photonic mesh decomposition algorithms.
- Replace numerical optimization of MZI phase parameters with closed-form analytical solutions.
- Reconstruct arbitrary unitary matrices using cascaded MZI structures.
- Compare the architectures in terms of optical depth, insertion loss, and phase-error sensitivity.
- Investigate how mesh performance scales with increasing matrix dimensions.
- Validate numerical accuracy using random unitary matrices and special matrix configurations.

## Mathematical Framework

### 1. Mach–Zehnder Interferometer (MZI)

The fundamental building block is a balanced directional coupler combined with two phase shifters.

The coupler matrix is defined as:

\[
B=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1&i\\
i&1
\end{bmatrix}
\]

The phase-shifter matrix is:

\[
P(\theta)=
\begin{bmatrix}
e^{i\theta}&0\\
0&1
\end{bmatrix}
\]

The MZI transfer matrix is:

\[
T(\theta,\phi)=BP(\theta)BP(\phi)
\]

where:
- \(\theta\): Internal phase controlling the power-splitting ratio.
- \(\phi\): External phase controlling the relative output phase.

### 2. Closed-Form Optical Elimination

One of the main improvements is replacing BFGS numerical optimization with a closed-form solution for determining MZI phase settings.

Given an input vector:

\[
\begin{bmatrix}
a\\b
\end{bmatrix}
\]

the objective is to determine the MZI parameters that eliminate one output component.

The analytical solution eliminates the need for an initial optimization guess and iterative numerical minimization.

An additional output phase correction ensures that the surviving amplitude is real and positive.

### 3. Unitary Matrix Decomposition

An arbitrary \(N\times N\) unitary matrix is represented using an output phase screen and a sequence of two-mode MZI transformations:

\[
U=D T_1T_2\cdots T_K
\]

where:

\[
K=\frac{N(N-1)}{2}
\]

and \(D\) represents the output phase shifts.

Two decomposition strategies are implemented:

- **Reck architecture:** Uses a triangular mesh arrangement.
- **Clements architecture:** Uses a rectangular mesh arrangement with alternating left and right elimination operations.

The Clements implementation also incorporates a closed-form 2×2 factorization to move diagonal phase operations through the MZI network.

## Implementation

The notebook is organized into the following sections:

| Section | Description |
|---|---|
| Optical building blocks | Coupler, phase shifter, and MZI transfer matrices |
| Closed-form elimination | Analytical MZI phase calculation and comparison with BFGS |
| 3×3 decomposition | Step-by-step elimination and matrix reconstruction |
| Reck vs. Clements | Implementation and comparison of 4×4 meshes |
| Insertion loss | Evaluation of transmission, fidelity, and reconstruction error |
| Phase-setting errors | Monte Carlo analysis of random MZI phase deviations |
| Mesh-size scaling | Performance evaluation for increasing matrix dimensions |
| Validation | Numerical accuracy tests using random and special unitary matrices |

## Performance Evaluation

### 1. Mesh Architecture Comparison

The two architectures are compared using:

- Number of MZIs.
- Optical depth.
- Matrix reconstruction error.
- Normalized matrix fidelity.

The theoretical optical depths are:

\[
D_{\mathrm{Reck}}=2N-3
\]

\[
D_{\mathrm{Clements}}=N
\]

Both architectures use the same number of MZIs, but their physical arrangements differ.

### 2. Insertion Loss Analysis

A configurable loss model is implemented to investigate the effects of optical component losses.

The model considers:
- Directional coupler insertion loss.
- Phase-shifter attenuation.
- Total coupler loss per MZI.

The analysis evaluates:

- Average optical power transmission.
- Normalized matrix fidelity.
- Matrix reconstruction error.
- Unitarity error.

The Clements architecture is particularly interesting because its more balanced optical paths can help preserve the reconstructed matrix under uniform component losses.

### 3. Monte Carlo Phase-Error Analysis

Random fabrication and phase-setting deviations are modeled using independent Gaussian perturbations applied to the internal and external MZI phases:

\[
\theta_k'=\theta_k+\epsilon_{\theta,k}
\]

\[
\phi_k'=\phi_k+\epsilon_{\phi,k}
\]

where the phase errors are sampled from Gaussian distributions with standard deviation \(\sigma\).

The simulation uses:

- 5,000 Monte Carlo trials.
- Multiple phase-error standard deviations.
- Identical random error samples for both architectures.
- Mean fidelity, standard deviation, and paired statistical comparison.

This provides a statistical comparison of the sensitivity of Reck and Clements meshes to phase-setting errors.

### 4. Scalability Analysis

The implementation investigates mesh performance for:

\[
N\in\{4,8,16,32\}
\]

The scaling study evaluates:

- Optical depth.
- Average transmission under insertion loss.
- Matrix fidelity under insertion loss.
- Matrix fidelity under random phase errors.

The analysis uses six Haar-random unitary matrices per mesh size, with 100 phase-noise trials per matrix.

## Algorithmic Improvements

Compared with the earlier implementations, this version introduces several improvements:

1. **Analytical MZI optimization:** Replaces BFGS-based phase optimization with a closed-form solution.
2. **Self-contained decomposition algorithms:** Implements both Reck and Clements architectures without requiring the external `pnn` package.
3. **Unified mesh representation:** Both architectures use the same data structure and reconstruction interface.
4. **Improved numerical stability:** Includes handling of degenerate cases and near-zero phase-related parameters.
5. **Vectorized simulations:** Supports batched matrix reconstruction for Monte Carlo analysis.
6. **Statistical comparison:** Uses common random numbers to reduce variance in paired architecture comparisons.
7. **Extended validation:** Tests random unitary matrices and special cases, including identity, permutation, and DFT matrices.

## Validation

The implementation includes numerical validation for:

- Random unitary matrices with dimensions \(N=2\) through \(N=16\).
- Twenty random matrices per dimension.
- Identity matrices.
- Cyclic-shift matrices.
- Pair-swap matrices.
- Anti-diagonal matrices.
- Discrete Fourier transform matrices.

Both decomposition algorithms are checked against their reconstructed matrices using the Frobenius norm:

\[
E=\|U-U_{\mathrm{reconstructed}}\|_F
\]

The notebook also includes regression checks against previously saved reference results for selected 4×4 simulations.

## Requirements

The project uses Python and the following libraries:

- NumPy
- Matplotlib
- SciPy (only for the BFGS comparison in the closed-form elimination section)

## Installation and Execution

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Install the required dependencies:

```bash
pip install numpy matplotlib scipy
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Photonic_Decomposition_Improved.ipynb
```

Run all cells sequentially to reproduce the decomposition algorithms, numerical comparisons, simulations, and plots.

**Note:** SciPy is only needed for the comparison between BFGS optimization and the analytical MZI solution. The decomposition algorithms themselves are implemented using NumPy.

## Technologies

- Python
- NumPy
- Matplotlib
- SciPy
- Numerical Linear Algebra
- Photonic Integrated Circuits
- Mach–Zehnder Interferometers
- Unitary Matrix Decomposition
- Monte Carlo Simulation

## Applications and Relevance

Photonic mesh architectures are important building blocks in programmable photonic integrated circuits and optical information-processing systems.

The decomposition methods investigated in this project are relevant to:

- Programmable photonic circuits.
- Optical matrix processing.
- Photonic neural networks.
- Interferometer-based optical computing.
- Optical signal processing.
- Scalable integrated photonic architectures.

## Project Scope

This project focuses on mathematical modeling, numerical decomposition, and simulation-based performance evaluation of idealized MZI meshes with configurable component losses and phase errors.

It does not include electromagnetic device-level simulation, fabrication, or experimental characterization of physical photonic components.

---

**Author:** [Your Name]

**Project:** Photonic Unitary Matrix Decomposition and Performance Analysis

