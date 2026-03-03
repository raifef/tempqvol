# QVolutionHackathon-NativePhotonicQRC
## Hybrid GRU + Photonic Memory (measurement-feedback PQRC)

This model is a **hybrid classical–photonic time-series forecaster**: a **GRU backbone** learns the smooth, low-frequency structure of the swaption surface dynamics, while a **photonic-inspired reservoir (“photonic memory”)** supplies high-dimensional, nonlinear, *fading-memory* features that are used to correct the GRU via a residual head.

### Key references (where the ideas come from)

- **Photonic encoding (phase modulation in a reconfigurable interferometer):**  
  *R. Di Bartolo et al.*, **“Time-series forecasting with multiphoton quantum states and integrated photonics”**, arXiv:2512.02928 (2025).  
  Core idea we borrow: **encode the input signal into optical phase(s)** of a reconfigurable linear-optical circuit; extract features from measurement statistics.

- **Photonic memory via measurement-conditioned feedback (recurrence without training internal weights):**  
  *Çağın Ekici*, **“A Programmable Linear Optical Quantum Reservoir with Measurement Feedback for Time Series Analysis”**, arXiv:2602.17440 (2026).  
  Core idea we borrow: introduce **memory/recurrence** by **feeding back** (a function of) measured features to update a *subset* of programmable phases, producing controllable fading memory and nonlinear temporal processing.
  Inspired by these two recent works.
Ekici, Çağın. "A Programmable Linear Optical Quantum Reservoir with Measurement Feedback for Time Series Analysis." arXiv preprint arXiv:2602.17440 (2026).
Di Bartolo, Rosario, et al. "Time-series forecasting with multiphoton quantum states and integrated photonics." arXiv preprint arXiv:2512.02928 (2025).

---

## Architecture overview

At each time step `t` (or each observation index), we maintain:

- `x_t`: classical input features (e.g., PCA factors of the vol surface, micro-features, calendar features, etc.)
- `r_t`: **photonic memory state**, produced by a photonic feature map with **feedback**
- `ŷ_t,h`: prediction for horizon `h` (multi-horizon forecasting)

High-level dataflow:


Where `α_h` is a **per-horizon gate** (often crucial in practice): it lets the model use strong correction on short horizons while suppressing harmful long-horizon residual corrections.

---

## Photonic encoding (what the “photonic memory” computes)

### 1) Encode classical inputs into photonic circuit parameters
We map the current input vector `x_t` into a set of circuit phases:

- `θ_in(t) = E(x_t)` where `E(·)` is a simple encoding map (typically linear + scaling + wrap/clamp into `[0, 2π)`).
- Practically, the encoded phases drive a reconfigurable interferometer / phase shifters.

This is the “temporal photonic encoding” / “phase modulation” concept: the time-series drives the photonic device *through its phases*.

### 2) Fixed interferometer + multiphoton feature readout
We propagate a (simulated) multiphoton input state through a fixed linear-optical network `U` and extract coarse-grained measurement features:

- single-click / marginal stats (if used)
- **two-photon coincidence features** (commonly the most informative and stable in our implementation)

We denote the extracted feature vector as:

- `φ_t = Φ(U, θ(t))`

where `φ_t` is a high-dimensional nonlinear embedding of the current input.

### 3) Measurement-conditioned feedback = “photonic memory”
To turn a feedforward photonic embedding into a **recurrent reservoir**, we add a feedback update that uses the previous reservoir output/state:

One convenient abstraction consistent with our implementation:

- Maintain an internal reservoir state `r_t` (or equivalently a subset of feedback phases).
- Update it using a leaky integration + feedback strength:


- `β` controls the fading-memory timescale (“leak rate”).
- `fb_strength` controls how strongly the previous state perturbs the next circuit configuration.
- `W_fb` is a fixed (often random) projection selecting a *budgeted subset* of phases to update.
- `g(·)` is a simple nonlinearity / clipping to keep the feedback stable.

**Intuition:** the circuit’s configuration at `t+1` depends on what it “saw” at `t`, creating recurrence and temporal feature mixing *without backprop training the photonic internals*.

---

## GRU backbone + residual correction

### GRU backbone
- Input: windowed sequence `x_{t-L+1:t}`
- Output: multi-horizon forecast `y_GRU(t, h)` (direct or seq2seq head)

The GRU learns the “easy” part: smooth temporal evolution, autocorrelation structure, and stable low-dimensional dynamics.

### Photonic residual head (fast linear readout)
Compute residual targets using the training data:

- `e(t, h) = y_true(t, h) - y_GRU(t, h)`

Train a simple readout (typically ridge regression) on the photonic memory state:

- `Δy(t, h) = W_out(h)^T · [1 ; r_t ; maybe x_t]`

This is cheap, stable, and aligns with reservoir computing practice: **train only the readout**, keep the reservoir fixed.

### Final prediction (with horizon gating)

`α_h` is tuned on validation (and can be forced to 0 for horizons where the residual head is noisy or anti-correlated).

---

## Training procedure we use (typical)

1. **Train GRU** on the training split to minimize RMSE (or MAE/RMSE combo).
2. **Freeze GRU**, compute residuals `e(t, h)` on train.
3. **Generate photonic memory states** `r_t` by running the photonic feature map with feedback through the sequence.
4. **Fit ridge regression** readouts `W_out(h)` to predict `e(t, h)` from `r_t`.
5. **Tune** `fb_strength`, ridge `λ`, leak `β`, and **per-horizon gates** `α_h` on validation.
6. Optionally: re-fit readout on train+val with chosen hyperparameters, then evaluate on test.

---

## What’s novel / “hackathon spin”

- **Photonic memory as a modular drop-in:** we treat a measurement-feedback photonic reservoir as an explicit *memory module* that augments a classical recurrent model.
- **Hardware-aligned features:** we specifically use **coincidence-style features** that map cleanly onto photonic detection primitives.
- **Recurrence without training the quantum block:** feedback introduces memory/temporal processing while keeping the photonic part essentially “fixed + programmable”, avoiding heavy gradient-based training through a quantum simulator.
- **Residualization + gating:** the photonic block only needs to explain what the GRU misses, and the gate `α_h` prevents long-horizon degradation—this is often the difference between “quantum hurts” and “quantum helps”.

---

## Implementation notes (as reflected in our codebase)

- The photonic block exposes knobs like:
  - `fb_strength` (feedback gain)
  - `ridge` (readout regularization)
  - feature choice including `two_photon_coinc_features`
- Metrics we track include RMSE/MAE (MAPE only when appropriate for strictly-positive targets).
- The model is designed to be **apples-to-apples** against classical baselines: same splits, same windowing, and the photonic memory is evaluated as an additive improvement over the GRU.



