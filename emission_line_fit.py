"""
emission_line_fit.py
====================
Bayesian emission-line fitting pipeline for H-alpha + NII doublet spectra.

Data format expected
--------------------
  spectra   : (N, M) float array   – flux per spectral channel
  ivar      : (N, M) float array   – inverse variance
  wavelength: (M,)   float array   – wavelength in Angstroms
  xy_coords : (N, 2) int/float arr – pixel (x, y) for each spectrum

Pipeline
--------
  1. Build a NumPyro model for a single pixel (3 tied Gaussian emission
     lines + low-order Legendre continuum).
  2. vmap the log-likelihood over all N pixels so a single NUTS chain
     advances every pixel simultaneously on the GPU.
  3. Run MCMC; save posterior samples.
  4. Post-process with ArviZ:
       • R-hat / ESS convergence diagnostics
       • 95 % HDI on line amplitudes  →  Tier-1 detection flag
       • WAIC model comparison        →  Tier-2 detection flag
  5. Write per-pixel summary table and 2-D detection / kinematic maps.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from spectral_cube import SpectralCube

import jax
jax.config.update("jax_enable_x64", True)   # must be set before any JAX ops
import jax.numpy as jnp
from jax import vmap, random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

import arviz as az

# ---------------------------------------------------------------------------
# 0.  Global configuration
# ---------------------------------------------------------------------------

# Speed of light (km/s)
C_KMS = 2.99792458e5

# Rest-frame wavelengths (Angstroms, vacuum)
LAMBDA_HA    = 6564.61
LAMBDA_NII_R = 6585.27   # the brighter NII line
LAMBDA_NII_B = 6549.86   # the fainter  NII line
NII_RATIO    = 1.0 / 3.0  # A_NII6548 / A_NII6583  (atomic physics)

# Fitting / MCMC settings
MCMC_CONFIG = dict(
    num_warmup  = 1000,
    num_samples = 1000,
    num_chains  = 1,          # vmap handles parallelism; 1 chain per pixel
    progress_bar= True,
)

# Detection thresholds
HDI_PROB          = 0.95   # credible interval for Tier-1
WAIC_DELTA_DETECT = 6.0    # ΔWAIC favouring line model for Tier-2


# ---------------------------------------------------------------------------
# 1.  Utility: wavelength ↔ velocity conversions
# ---------------------------------------------------------------------------

def wave_to_vel(wave: jnp.ndarray, wave_rest: float) -> jnp.ndarray:
    """Observed wavelength → recession velocity (km/s)."""
    return (wave / wave_rest - 1.0) * C_KMS


def vel_to_wave(v_kms: float, wave_rest: float) -> float:
    """Recession velocity (km/s) → observed wavelength."""
    return wave_rest * (1.0 + v_kms / C_KMS)


# ---------------------------------------------------------------------------
# 2.  Gaussian profile (JAX-differentiable)
# ---------------------------------------------------------------------------

def gaussian(wave: jnp.ndarray, amp: float, center: float, sigma: float) -> jnp.ndarray:
    """Unnormalised Gaussian emission-line profile."""
    return amp * jnp.exp(-0.5 * ((wave - center) / sigma) ** 2)


# ---------------------------------------------------------------------------
# 3.  Full spectral model
#       free params: v_los, sigma_v, A_Ha, A_NII, c0, c1
# ---------------------------------------------------------------------------

def spectral_model(
    wave:    jnp.ndarray,   # (M,)
    v_los:   float,         # km/s  – line-of-sight recession velocity
    sigma_v: float,         # km/s  – velocity dispersion (intrinsic + inst.)
    A_Ha:    float,         # amplitude of H-alpha
    A_NII:   float,         # amplitude of NII 6583
    c0:      float,         # continuum intercept (at wave_mid)
    c1:      float,         # continuum slope
) -> jnp.ndarray:
    """
    Compute model flux at each wavelength channel.

    Continuum: linear baseline  c0 + c1*(wave - wave_mid)
    Lines    : H-alpha + NII doublet, kinematics tied, NII ratio fixed.
    """
    wave_mid = 0.5 * (wave[0] + wave[-1])
    continuum = c0 + c1 * (wave - wave_mid)

    # Observed line centres from recession velocity
    lam_Ha   = vel_to_wave(v_los, LAMBDA_HA)
    lam_NIIR = vel_to_wave(v_los, LAMBDA_NII_R)
    lam_NIIB = vel_to_wave(v_los, LAMBDA_NII_B)

    # Convert velocity dispersion to Angstrom sigma for each line
    sig_Ha   = lam_Ha   * sigma_v / C_KMS
    sig_NIIR = lam_NIIR * sigma_v / C_KMS
    sig_NIIB = lam_NIIB * sigma_v / C_KMS

    lines = (
        gaussian(wave, A_Ha,              lam_Ha,   sig_Ha)
      + gaussian(wave, A_NII,             lam_NIIR, sig_NIIR)
      + gaussian(wave, A_NII * NII_RATIO, lam_NIIB, sig_NIIB)
    )

    return continuum + lines


def _estimate_initial_params(
    wave:           np.ndarray,
    flux:           np.ndarray,
    ivar:           np.ndarray,
    v_center:       float,
    v_half_range:   float,
    sigma_inst_kms: float,
) -> dict:
    """
    Estimate starting parameter values from the data using peak-finding.

    Finds the brightest channel within the Hα prior window and converts it
    to a velocity.  Used to initialise NUTS via init_to_value so the chain
    starts near the correct mode rather than a random prior draw.

    Returns a dict suitable for passing to init_to_value().
    """
    # Wavelength range corresponding to the v_los prior window for Hα
    ha_lo = vel_to_wave(v_center - v_half_range * 0.95, LAMBDA_HA)
    ha_hi = vel_to_wave(v_center + v_half_range * 0.95, LAMBDA_HA)
    ha_mask = (wave >= ha_lo) & (wave <= ha_hi) & (ivar > 0)

    if ha_mask.sum() < 3 or np.nanmax(flux[ha_mask]) <= 0:
        # fallback: prior centre with instrumental sigma
        return dict(
            v_los   = float(v_center),
            sigma_v = float(sigma_inst_kms * 1.5),
            A_Ha    = float(np.nanpercentile(np.abs(flux), 90)),
            A_NII   = float(np.nanpercentile(np.abs(flux), 75)),
            c0      = 0.0,
            c1      = 0.0,
        )

    # Smooth flux with a 3-channel box to reduce noise sensitivity
    flux_sm = np.convolve(flux * (ivar > 0), np.ones(3) / 3, mode="same")

    # v_los from peak wavelength within the window
    peak_idx  = np.argmax(flux_sm * ha_mask)
    peak_wave = wave[peak_idx]
    v_init    = float(wave_to_vel(peak_wave, LAMBDA_HA))
    # Clamp slightly inside the prior to avoid boundary issues
    v_init    = float(np.clip(v_init,
                              v_center - v_half_range * 0.9,
                              v_center + v_half_range * 0.9))

    # sigma_v from the second moment of flux in the window around the peak
    window = ha_mask & (np.abs(wave - peak_wave) < 15.0)   # ±15 Å
    if window.sum() > 2:
        w     = np.maximum(flux_sm[window], 0)
        dv    = (wave[window] - peak_wave) / peak_wave * C_KMS
        var_v = np.sum(w * dv**2) / (np.sum(w) + 1e-30)
        sig_v = float(np.sqrt(np.maximum(var_v, sigma_inst_kms**2)))
        sig_v = float(np.clip(sig_v, sigma_inst_kms * 1.01, 250.0))
    else:
        sig_v = float(sigma_inst_kms * 1.5)

    # Amplitudes from peak flux near each line centre
    A_Ha  = float(np.nanmax(flux_sm[ha_mask]))
    # NII 6583 region
    nii_lo   = vel_to_wave(v_init - 200, LAMBDA_NII_R)
    nii_hi   = vel_to_wave(v_init + 200, LAMBDA_NII_R)
    nii_mask = (wave >= nii_lo) & (wave <= nii_hi) & (ivar > 0)
    A_NII    = float(np.nanmax(flux_sm[nii_mask])) if nii_mask.sum() > 0 else A_Ha / 3.0
    A_NII    = max(A_NII, 1e-3)

    return dict(
        v_los   = v_init,
        sigma_v = sig_v,
        A_Ha    = max(A_Ha,  1e-3),
        A_NII   = A_NII,
        c0      = float(np.nanmedian(flux)),
        c1      = 0.0,
    )


# ---------------------------------------------------------------------------
# 4.  NumPyro models  (line model  &  continuum-only null model)
# ---------------------------------------------------------------------------

def make_priors(wave, a_ha_scale, a_nii_scale, continuum_scale,
                v_center, v_half_range, sigma_inst):
    """Return a dict of prior distributions tuned to the data."""
    return dict(
        v_los   = dist.Uniform(v_center - v_half_range, v_center + v_half_range),
        sigma_v = dist.Uniform(sigma_inst, 300.0),          # km/s
        A_Ha    = dist.HalfNormal(a_ha_scale),
        A_NII   = dist.HalfNormal(a_nii_scale),
        c0      = dist.Normal(0.0, continuum_scale),
        c1      = dist.Normal(0.0, continuum_scale),
    )


def line_model(wave, flux, ivar, priors):
    """
    NumPyro model: continuum + H-alpha + NII doublet.
    Likelihood: Gaussian with variance = 1/ivar (per-channel).
    """
    v_los   = numpyro.sample("v_los",   priors["v_los"])
    sigma_v = numpyro.sample("sigma_v", priors["sigma_v"])
    A_Ha    = numpyro.sample("A_Ha",    priors["A_Ha"])
    A_NII   = numpyro.sample("A_NII",   priors["A_NII"])
    c0      = numpyro.sample("c0",      priors["c0"])
    c1      = numpyro.sample("c1",      priors["c1"])

    mu = spectral_model(wave, v_los, sigma_v, A_Ha, A_NII, c0, c1)

    # Mask channels with zero / negative ivar
    safe_ivar = jnp.where(ivar > 0, ivar, 1e-30)
    sigma_obs = 1.0 / jnp.sqrt(safe_ivar)
    obs_mask  = ivar > 0

    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("obs", dist.Normal(mu, sigma_obs), obs=flux)


def continuum_model(wave, flux, ivar, priors):
    """
    Null NumPyro model: continuum only (no emission lines).
    Used for WAIC model comparison.
    """
    c0 = numpyro.sample("c0", priors["c0"])
    c1 = numpyro.sample("c1", priors["c1"])

    wave_mid  = 0.5 * (wave[0] + wave[-1])
    mu        = c0 + c1 * (wave - wave_mid)

    safe_ivar = jnp.where(ivar > 0, ivar, 1e-30)
    sigma_obs = 1.0 / jnp.sqrt(safe_ivar)
    obs_mask  = ivar > 0

    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("obs", dist.Normal(mu, sigma_obs), obs=flux)


# ---------------------------------------------------------------------------
# 5.  Batched (vmapped) MCMC over all N pixels
#
#     NumPyro does not directly vmap NUTS over pixels, so we use the
#     recommended pattern: pass the full (N, M) arrays as plate-free
#     batched data and rely on JAX's vmap inside a custom potential_fn.
#
#     For maximum flexibility we loop over pixels but JIT the inner kernel
#     and use device parallelism.  For N > ~5 000 on a modern GPU you may
#     prefer to split into mini-batches.
# ---------------------------------------------------------------------------

def run_mcmc_all_pixels(
    wave:    np.ndarray,   # (M,)
    spectra: np.ndarray,   # (N, M)
    ivar:    np.ndarray,   # (N, M)
    priors:  dict,
    rng_key: jax.Array,
    mcmc_cfg: dict = MCMC_CONFIG,
    batch_size: int = 512,
) -> dict:
    """
    Run NUTS MCMC for every pixel with data-driven initialisation.

    Per-pixel initial parameter estimates are computed from the data using
    _estimate_initial_params before any JAX work begins.  They are passed
    as batched arguments into the vmapped run_one function and used via
    init_to_value, so each pixel's chain starts near its own likelihood
    peak rather than at a random prior draw.

    Returns
    -------
    samples : dict  { param_name : (N, num_samples) array }
    """
    N = spectra.shape[0]
    wave_j    = jnp.array(wave)
    spectra_j = jnp.array(spectra)
    ivar_j    = jnp.array(ivar)

    # ---- extract prior bounds needed by _estimate_initial_params -----------
    v_center       = float((priors["v_los"].low  + priors["v_los"].high) / 2)
    v_half_range   = float((priors["v_los"].high - priors["v_los"].low)  / 2)
    sigma_inst_kms = float(priors["sigma_v"].low)

    # ---- compute per-pixel initial params (fast NumPy loop, pre-JAX) -------
    print("  Computing data-driven initial parameters for all pixels...")
    init_list = [
        _estimate_initial_params(
            wave           = wave,
            flux           = spectra[i],
            ivar           = ivar[i],
            v_center       = v_center,
            v_half_range   = v_half_range,
            sigma_inst_kms = sigma_inst_kms,
        )
        for i in range(N)
    ]
    # Stack into (N,) JAX arrays, one per parameter
    init_j = {
        k: jnp.array([ip[k] for ip in init_list])
        for k in ["v_los", "sigma_v", "A_Ha", "A_NII", "c0", "c1"]
    }

    # ---- single-pixel kernel with per-pixel init_to_value ------------------
    def run_one(flux_i, ivar_i, key_i,
                init_v, init_s, init_ha, init_nii, init_c0, init_c1):
        init_vals = {
            "v_los":   init_v,
            "sigma_v": init_s,
            "A_Ha":    init_ha,
            "A_NII":   init_nii,
            "c0":      init_c0,
            "c1":      init_c1,
        }
        kernel = NUTS(line_model,
                      init_strategy=init_to_value(values=init_vals))
        mcmc   = MCMC(
            kernel,
            num_warmup  = mcmc_cfg["num_warmup"],
            num_samples = mcmc_cfg["num_samples"],
            num_chains  = mcmc_cfg["num_chains"],
            progress_bar= False,
        )
        mcmc.run(key_i, wave_j, flux_i, ivar_i, priors)
        return mcmc.get_samples()

    batched_run = vmap(run_one)

    all_samples = {k: [] for k in ["v_los", "sigma_v", "A_Ha", "A_NII", "c0", "c1"]}

    n_batches = int(np.ceil(N / batch_size))
    for b in range(n_batches):
        sl   = slice(b * batch_size, min((b + 1) * batch_size, N))
        keys = random.split(rng_key, spectra_j[sl].shape[0])
        samps = batched_run(
            spectra_j[sl], ivar_j[sl], keys,
            init_j["v_los"][sl],
            init_j["sigma_v"][sl],
            init_j["A_Ha"][sl],
            init_j["A_NII"][sl],
            init_j["c0"][sl],
            init_j["c1"][sl],
        )
        for k, v in samps.items():
            all_samples[k].append(np.array(v))

        print(f"  Batch {b+1}/{n_batches} done  ({sl.stop} / {N} pixels)")

    return {k: np.concatenate(v, axis=0) for k, v in all_samples.items()}


# ---------------------------------------------------------------------------
# 6.  Null-model MCMC (continuum only) – used for WAIC comparison
# ---------------------------------------------------------------------------

def run_mcmc_continuum(
    wave:    np.ndarray,
    spectra: np.ndarray,
    ivar:    np.ndarray,
    priors:  dict,
    rng_key: jax.Array,
    mcmc_cfg: dict = MCMC_CONFIG,
    batch_size: int = 512,
) -> dict:
    """Same as run_mcmc_all_pixels but for the continuum-only null model.

    Initialises c0 to the per-pixel median flux and c1 to zero.
    """
    N         = spectra.shape[0]
    wave_j    = jnp.array(wave)
    spectra_j = jnp.array(spectra)
    ivar_j    = jnp.array(ivar)

    # Per-pixel continuum init: c0 = median flux, c1 = 0
    init_c0_j = jnp.array(np.nanmedian(spectra, axis=1))
    init_c1_j = jnp.zeros(N)

    def run_one(flux_i, ivar_i, key_i, init_c0, init_c1):
        init_vals = {"c0": init_c0, "c1": init_c1}
        kernel = NUTS(continuum_model,
                      init_strategy=init_to_value(values=init_vals))
        mcmc   = MCMC(
            kernel,
            num_warmup  = mcmc_cfg["num_warmup"],
            num_samples = mcmc_cfg["num_samples"],
            num_chains  = mcmc_cfg["num_chains"],
            progress_bar= False,
        )
        mcmc.run(key_i, wave_j, flux_i, ivar_i, priors)
        return mcmc.get_samples()

    batched_run = vmap(run_one)
    all_samples = {"c0": [], "c1": []}

    n_batches = int(np.ceil(N / batch_size))
    for b in range(n_batches):
        sl   = slice(b * batch_size, min((b + 1) * batch_size, N))
        keys = random.split(rng_key, spectra_j[sl].shape[0])
        samps = batched_run(
            spectra_j[sl], ivar_j[sl], keys,
            init_c0_j[sl], init_c1_j[sl],
        )
        for k, v in samps.items():
            all_samples[k].append(np.array(v))

    return {k: np.concatenate(v, axis=0) for k, v in all_samples.items()}

    return {k: np.concatenate(v, axis=0) for k, v in all_samples.items()}


# ---------------------------------------------------------------------------
# 7.  Post-processing: diagnostics + detection flags
# ---------------------------------------------------------------------------

def compute_hdi_detection(samples_A: np.ndarray, hdi_prob: float = HDI_PROB) -> np.ndarray:
    """
    Tier-1 detection: amplitude HDI lower bound > 0.

    Parameters
    ----------
    samples_A : (N, S) array of posterior amplitude samples
    hdi_prob  : credible interval probability

    Returns
    -------
    detected  : (N,) bool array
    hdi_lo    : (N,) lower HDI bound
    hdi_hi    : (N,) upper HDI bound
    """
    N = samples_A.shape[0]
    hdi_lo = np.zeros(N)
    hdi_hi = np.zeros(N)

    for i in range(N):
        interval = az.hdi(samples_A[i], hdi_prob=hdi_prob)
        hdi_lo[i] = interval[0]
        hdi_hi[i] = interval[1]

    detected = hdi_lo > 0
    return detected, hdi_lo, hdi_hi


def compute_rhat(samples: dict) -> dict:
    """
    Compute per-pixel R-hat for each parameter using ArviZ.
    samples[param] shape: (N, S)   (single chain – R-hat requires ≥2 chains
    for a proper estimate; here we split the chain in half as a proxy.)

    ArviZ from_dict expects shape (chains, draws) for scalar parameters.
    We pass (2, half) directly — NOT (1, 2, half), which ArviZ would
    misinterpret as 1 chain × 2 draws of a length-half vector.

    Returns dict { param: (N,) float array }.
    """
    N    = next(iter(samples.values())).shape[0]
    rhat = {k: np.full(N, np.nan) for k in samples}

    for i in range(N):
        posterior_dict = {}
        for k, v in samples.items():
            s    = v[i]          # (S,)
            S    = s.shape[0]
            half = S // 2
            if half < 2:
                continue         # too few samples; leave as NaN
            # shape (2, half) = (2 pseudo-chains, half draws) ← correct
            posterior_dict[k] = s[: 2 * half].reshape(2, half)

        if not posterior_dict:
            continue

        try:
            idata   = az.from_dict(posterior=posterior_dict)
            summary = az.summary(idata, var_names=list(posterior_dict.keys()),
                                 round_to=4)
            for k in posterior_dict:
                if k in summary.index:
                    rhat[k][i] = summary.loc[k, "r_hat"]
                else:
                    # fall back to first row whose index starts with the param name
                    matches = [idx for idx in summary.index
                               if idx == k or idx.startswith(f"{k}[")]
                    if matches:
                        rhat[k][i] = summary.loc[matches[0], "r_hat"]
        except Exception:
            pass   # leave as NaN for this pixel

    return rhat


def compute_waic_detection(
    wave:          np.ndarray,
    spectra:       np.ndarray,
    ivar:          np.ndarray,
    line_samples:  dict,
    cont_samples:  dict,
    priors:        dict,
    rng_key:       jax.Array,
    delta_thresh:  float = WAIC_DELTA_DETECT,
) -> np.ndarray:
    """
    Tier-2 detection via WAIC model comparison.

    Computes WAIC for the line model and the continuum-only model for each
    pixel.  Returns a boolean array: True where ΔWAIC (line − continuum) < 0
    (i.e. the line model has lower WAIC, meaning better predictive accuracy)
    AND |ΔWAIC| > delta_thresh.

    Parameters
    ----------
    line_samples : dict  { param: (N, S) }  – from run_mcmc_all_pixels
    cont_samples : dict  { param: (N, S) }  – from run_mcmc_continuum

    Returns
    -------
    waic_detected : (N,) bool
    delta_waic    : (N,) float   (negative = line model preferred)
    """
    N      = spectra.shape[0]
    wave_j = jnp.array(wave)
    delta_waic    = np.zeros(N)
    waic_detected = np.zeros(N, dtype=bool)

    for i in range(N):
        flux_i = jnp.array(spectra[i])
        ivar_i = jnp.array(ivar[i])

        # --- Line model WAIC -------------------------------------------------
        ll_line = _pixel_log_likelihood_line(
            wave_j, flux_i, ivar_i, line_samples, i
        )
        post_line = {k: v[i][np.newaxis, :]   # (1 chain, S draws)
                     for k, v in line_samples.items()}
        idata_line = az.from_dict(
            posterior     = post_line,
            log_likelihood= {"obs": ll_line},   # (1, S)
        )

        # --- Continuum model WAIC --------------------------------------------
        ll_cont = _pixel_log_likelihood_cont(
            wave_j, flux_i, ivar_i, cont_samples, i
        )
        post_cont = {k: v[i][np.newaxis, :]
                     for k, v in cont_samples.items()}
        idata_cont = az.from_dict(
            posterior     = post_cont,
            log_likelihood= {"obs": ll_cont},   # (1, S)
        )

        try:
            w_line = az.waic(idata_line)
            w_cont = az.waic(idata_cont)
            dw     = float(w_line.elpd_waic) - float(w_cont.elpd_waic)
            # WAIC is on the elpd scale (higher = better), so line preferred if dw > 0
            delta_waic[i]    = dw
            waic_detected[i] = dw > delta_thresh
        except Exception:
            delta_waic[i]    = 0.0
            waic_detected[i] = False

    return waic_detected, delta_waic


def _pixel_log_likelihood_line(wave_j, flux_i, ivar_i, line_samples, i):
    """
    Compute per-draw log-likelihood array for the line model at pixel i.

    Returns shape (1, S) — summed over spectral channels — which is what
    az.from_dict expects for a scalar observation site: (chains, draws).
    """
    S = line_samples["v_los"].shape[1]
    log_likes = np.zeros((1, S))   # (1 pseudo-chain, S draws)

    safe_ivar = jnp.where(ivar_i > 0, ivar_i, 1e-30)
    sigma_obs = 1.0 / jnp.sqrt(safe_ivar)
    mask      = ivar_i > 0

    for s in range(S):
        mu = spectral_model(
            wave_j,
            float(line_samples["v_los"][i, s]),
            float(line_samples["sigma_v"][i, s]),
            float(line_samples["A_Ha"][i, s]),
            float(line_samples["A_NII"][i, s]),
            float(line_samples["c0"][i, s]),
            float(line_samples["c1"][i, s]),
        )
        ll = dist.Normal(mu, sigma_obs).log_prob(flux_i)
        ll = jnp.where(mask, ll, 0.0)
        log_likes[0, s] = float(jnp.sum(ll))   # sum over channels → scalar

    return log_likes


def _pixel_log_likelihood_cont(wave_j, flux_i, ivar_i, cont_samples, i):
    """
    Compute per-draw log-likelihood array for the continuum model at pixel i.

    Returns shape (1, S) — summed over spectral channels.
    """
    S        = cont_samples["c0"].shape[1]
    wave_mid = 0.5 * (wave_j[0] + wave_j[-1])
    log_likes = np.zeros((1, S))

    safe_ivar = jnp.where(ivar_i > 0, ivar_i, 1e-30)
    sigma_obs = 1.0 / jnp.sqrt(safe_ivar)
    mask      = ivar_i > 0

    for s in range(S):
        mu = cont_samples["c0"][i, s] + cont_samples["c1"][i, s] * (wave_j - wave_mid)
        ll = dist.Normal(mu, sigma_obs).log_prob(flux_i)
        ll = jnp.where(mask, ll, 0.0)
        log_likes[0, s] = float(jnp.sum(ll))

    return log_likes


# ---------------------------------------------------------------------------
# 8.  Build output maps
# ---------------------------------------------------------------------------

def build_maps(
    xy_coords:    np.ndarray,   # (N, 2)
    samples:      dict,         # { param: (N, S) }
    detected_t1:  np.ndarray,   # (N,) bool  – Tier-1
    detected_t2:  np.ndarray,   # (N,) bool  – Tier-2
    hdi_lo:       np.ndarray,   # (N,)
    hdi_hi:       np.ndarray,   # (N,)
    rhat:         dict,         # { param: (N,) }
) -> dict:
    """
    Assemble 2-D maps (on the native pixel grid) for key quantities.

    Returns a dict of 2-D arrays keyed by:
      detected_t1, detected_t2, detected_combined,
      v_los_med, v_los_std, sigma_v_med, sigma_v_std,
      A_Ha_med,  A_Ha_std,  A_NII_med,  A_NII_std,
      hdi_lo_Ha, hdi_hi_Ha,
      rhat_v_los, rhat_sigma_v, rhat_A_Ha
    """
    xs = xy_coords[:, 0].astype(int)
    ys = xy_coords[:, 1].astype(int)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    Nx = x1 - x0 + 1
    Ny = y1 - y0 + 1

    def empty():
        a = np.full((Ny, Nx), np.nan)
        return a

    maps = {k: empty() for k in [
        "detected_t1", "detected_t2", "detected_combined",
        "v_los_med", "v_los_std",
        "sigma_v_med", "sigma_v_std",
        "A_Ha_med", "A_Ha_std",
        "A_NII_med", "A_NII_std",
        "hdi_lo_Ha", "hdi_hi_Ha",
        "rhat_v_los", "rhat_sigma_v", "rhat_A_Ha",
    ]}

    xi = xs - x0
    yi = ys - y0

    maps["detected_t1"][yi, xi]       = detected_t1.astype(float)
    maps["detected_t2"][yi, xi]       = detected_t2.astype(float)
    maps["detected_combined"][yi, xi] = (detected_t1 & detected_t2).astype(float)

    for param, key_med, key_std in [
        ("v_los",   "v_los_med",   "v_los_std"),
        ("sigma_v", "sigma_v_med", "sigma_v_std"),
        ("A_Ha",    "A_Ha_med",    "A_Ha_std"),
        ("A_NII",   "A_NII_med",   "A_NII_std"),
    ]:
        maps[key_med][yi, xi] = np.median(samples[param], axis=1)
        maps[key_std][yi, xi] = np.std(samples[param],    axis=1)

    maps["hdi_lo_Ha"][yi, xi]    = hdi_lo
    maps["hdi_hi_Ha"][yi, xi]    = hdi_hi
    maps["rhat_v_los"][yi, xi]   = rhat["v_los"]
    maps["rhat_sigma_v"][yi, xi] = rhat["sigma_v"]
    maps["rhat_A_Ha"][yi, xi]    = rhat["A_Ha"]

    return maps


# ---------------------------------------------------------------------------
# 9.  Visualisation
# ---------------------------------------------------------------------------

def plot_maps(maps: dict, outdir: str = "."):
    """Save a multi-panel diagnostic figure."""
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle("Emission Line Fit Results", fontsize=14, y=1.01)

    panel_defs = [
        # (key,              title,                         cmap,          vmin,   vmax)
        ("detected_combined","Detection map (T1 ∩ T2)",     "Blues",       0,      1),
        ("v_los_med",        "v_los  [km/s]  (median)",     "RdBu_r",      None,   None),
        ("v_los_std",        "v_los  uncertainty [km/s]",   "YlOrRd",      0,      None),
        ("sigma_v_med",      "σ_v  [km/s]  (median)",       "viridis",     0,      None),
        ("sigma_v_std",      "σ_v  uncertainty [km/s]",     "YlOrRd",      0,      None),
        ("A_Ha_med",         "A(Hα)  (median)",             "inferno",     0,      None),
        ("A_Ha_std",         "A(Hα)  uncertainty",          "YlOrRd",      0,      None),
        ("rhat_v_los",       "R̂  v_los  (< 1.05 = OK)",    "RdYlGn_r",   1.0,    1.2),
        ("rhat_A_Ha",        "R̂  A(Hα)  (< 1.05 = OK)",   "RdYlGn_r",   1.0,    1.2),
    ]

    for ax, (key, title, cmap, vmin, vmax) in zip(axes.flat, panel_defs):
        data = maps[key]
        im   = ax.imshow(data, origin="lower", cmap=cmap,
                         vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [pix]", fontsize=7)
        ax.set_ylabel("y [pix]", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fname = os.path.join(outdir, "emission_line_maps.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved map figure → {fname}")
    plt.close()


def plot_example_fits(
    wave:     np.ndarray,
    spectra:  np.ndarray,
    ivar:     np.ndarray,
    samples:  dict,
    indices:  list,
    outdir:   str = ".",
):
    """
    Plot posterior predictive check for a handful of pixels.
    Draws 50 model realisations from the posterior for each selected pixel.
    """
    os.makedirs(outdir, exist_ok=True)
    wave_j = jnp.array(wave)
    n      = len(indices)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        flux     = spectra[idx]
        err      = np.where(ivar[idx] > 0, 1.0 / np.sqrt(ivar[idx]), np.nan)
        S        = samples["v_los"].shape[1]
        draw_idx = np.random.choice(S, size=min(50, S), replace=False)

        for s in draw_idx:
            mu = spectral_model(
                wave_j,
                float(samples["v_los"][idx, s]),
                float(samples["sigma_v"][idx, s]),
                float(samples["A_Ha"][idx, s]),
                float(samples["A_NII"][idx, s]),
                float(samples["c0"][idx, s]),
                float(samples["c1"][idx, s]),
            )
            ax.plot(wave, np.array(mu), color="steelblue", alpha=0.15, lw=0.8)

        ax.errorbar(wave, flux, yerr=err, fmt="k.", ms=2, elinewidth=0.7,
                    label="data", zorder=5)
        ax.set_title(f"Pixel {idx}", fontsize=9)
        ax.set_xlabel("λ [Å]")
        ax.set_ylabel("Flux")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fname = os.path.join(outdir, "example_fits.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved example fit figure → {fname}")
    plt.close()



# ---------------------------------------------------------------------------
# 10.  Save results
# ---------------------------------------------------------------------------

def save_results(
    outdir:      str,
    maps:        dict,
    samples:     dict,
    xy_coords:   np.ndarray,
    detected_t1: np.ndarray,
    detected_t2: np.ndarray,
    hdi_lo:      np.ndarray,
    hdi_hi:      np.ndarray,
    delta_waic:  np.ndarray,
    rhat:        dict,
):
    """Write per-pixel summary table and 2-D FITS maps."""
    os.makedirs(outdir, exist_ok=True)

    # --- Per-pixel ASCII summary ---
    header = (
        "# idx  x  y  detected_t1  detected_t2  "
        "v_los_med  v_los_std  sigma_v_med  sigma_v_std  "
        "A_Ha_med  A_Ha_std  hdi_lo_Ha  hdi_hi_Ha  "
        "delta_waic  rhat_vlos  rhat_Aha\n"
    )
    N = xy_coords.shape[0]
    rows = []
    for i in range(N):
        rows.append(
            f"{i:6d}  {xy_coords[i,0]:6.1f}  {xy_coords[i,1]:6.1f}  "
            f"{int(detected_t1[i]):3d}  {int(detected_t2[i]):3d}  "
            f"{np.median(samples['v_los'][i]):9.2f}  {np.std(samples['v_los'][i]):9.2f}  "
            f"{np.median(samples['sigma_v'][i]):9.2f}  {np.std(samples['sigma_v'][i]):9.2f}  "
            f"{np.median(samples['A_Ha'][i]):10.4e}  {np.std(samples['A_Ha'][i]):10.4e}  "
            f"{hdi_lo[i]:10.4e}  {hdi_hi[i]:10.4e}  "
            f"{delta_waic[i]:10.3f}  {rhat['v_los'][i]:.4f}  {rhat['A_Ha'][i]:.4f}\n"
        )
    summary_file = os.path.join(outdir, "pixel_summary.txt")
    with open(summary_file, "w") as f:
        f.write(header)
        f.writelines(rows)
    print(f"Saved pixel summary → {summary_file}")

    # --- 2-D FITS maps ---
    hdu_list = [fits.PrimaryHDU()]
    for key, data in maps.items():
        hdu = fits.ImageHDU(data=data.astype(np.float32), name=key.upper())
        hdu_list.append(hdu)
    fits_path = os.path.join(outdir, "emission_maps.fits")
    fits.HDUList(hdu_list).writeto(fits_path, overwrite=True)
    print(f"Saved FITS maps      → {fits_path}")


def diagnose_pixel(
    idx:       int,
    wave:      np.ndarray,
    spectra:   np.ndarray,
    ivar:      np.ndarray,
    samples:   dict,
    outdir:    str = ".",
):
    """
    Diagnostic plots and summary for a single pixel.

    Produces two figures:
      1. posterior_pixel_{idx}.png  – marginal posterior histograms for all
                                      6 parameters, with median marked.
      2. fit_pixel_{idx}.png        – data with error bars, 50 posterior
                                      draw realisations (blue), and the
                                      median model (red).

    Also prints a concise parameter summary table to stdout.
    """
    os.makedirs(outdir, exist_ok=True)
    wave_j = jnp.array(wave)
    params = ["v_los", "sigma_v", "A_Ha", "A_NII", "c0", "c1"]

    # ---- 1. Posterior summary table ----------------------------------------
    print(f"\n{'='*64}")
    print(f"  Pixel {idx} — posterior summary")
    print(f"{'='*64}")
    print(f"  {'param':>8}  {'median':>10}  {'std':>10}  "
          f"{'5th %':>10}  {'95th %':>10}")
    print(f"  {'-'*58}")
    for p in params:
        s = samples[p][idx]
        print(f"  {p:>8}  {np.median(s):>10.3f}  {np.std(s):>10.3f}  "
              f"  {np.percentile(s, 5):>10.3f}  {np.percentile(s, 95):>10.3f}")
    print(f"{'='*64}\n")

    # ---- 2. Posterior histogram panel --------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f"Pixel {idx} — posterior distributions", fontsize=11)

    for ax, p in zip(axes.flat, params):
        s = samples[p][idx]
        ax.hist(s, bins=40, color="steelblue", alpha=0.75, edgecolor="none")
        med = np.median(s)
        ax.axvline(med, color="red", lw=1.5, label=f"median={med:.2f}")
        ax.axvline(np.percentile(s,  5), color="orange", lw=1, ls="--",
                   label="5/95th pct")
        ax.axvline(np.percentile(s, 95), color="orange", lw=1, ls="--")
        ax.set_title(p, fontsize=9)
        ax.legend(fontsize=7)

    plt.tight_layout()
    hist_path = os.path.join(outdir, f"posterior_pixel_{idx}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {hist_path}")
    plt.close()

    # ---- 3. Spectral fit panel ---------------------------------------------
    flux = spectra[idx]
    err  = np.where(ivar[idx] > 0, 1.0 / np.sqrt(ivar[idx]), np.nan)
    S    = samples["v_los"].shape[1]

    mu_med = spectral_model(
        wave_j,
        float(np.median(samples["v_los"][idx])),
        float(np.median(samples["sigma_v"][idx])),
        float(np.median(samples["A_Ha"][idx])),
        float(np.median(samples["A_NII"][idx])),
        float(np.median(samples["c0"][idx])),
        float(np.median(samples["c1"][idx])),
    )

    draw_idx = np.random.choice(S, size=min(50, S), replace=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    for s in draw_idx:
        mu = spectral_model(
            wave_j,
            float(samples["v_los"][idx, s]),
            float(samples["sigma_v"][idx, s]),
            float(samples["A_Ha"][idx, s]),
            float(samples["A_NII"][idx, s]),
            float(samples["c0"][idx, s]),
            float(samples["c1"][idx, s]),
        )
        ax.plot(wave, np.array(mu), color="steelblue", alpha=0.15, lw=0.8)

    ax.plot(wave, np.array(mu_med), color="red", lw=2, label="median model",
            zorder=4)
    ax.errorbar(wave, flux, yerr=err, fmt="k.", ms=3, elinewidth=0.8,
                label="data", zorder=5)

    v_med = float(np.median(samples["v_los"][idx]))
    for lam_rest, label in [
        (LAMBDA_HA,    "Hα"),
        (LAMBDA_NII_R, "NII 6583"),
        (LAMBDA_NII_B, "NII 6548"),
    ]:
        lam_obs = vel_to_wave(v_med, lam_rest)
        ax.axvline(lam_obs, color="gray", lw=0.8, ls=":")
        ax.text(lam_obs, ax.get_ylim()[1], label, fontsize=7,
                ha="center", va="bottom", color="gray")

    ax.set_xlabel("λ [Å]")
    ax.set_ylabel("Flux")
    ax.set_title(f"Pixel {idx} — spectral fit", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fit_path = os.path.join(outdir, f"fit_pixel_{idx}.png")
    plt.savefig(fit_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {fit_path}")
    plt.close()


def run_single_pixel_diagnostic(
    pixel_idx:      int,
    spectra:        np.ndarray,
    ivar:           np.ndarray,
    wavelength:     np.ndarray,
    xy_coords:      np.ndarray,
    galaxy_v_kms:   float = 0.0,
    v_half_range:   float = 400.0,
    sigma_inst_kms: float = 50.0,
    a_ha_scale:     float = None,
    a_nii_scale:    float = None,
    cont_scale:     float = None,
    num_warmup:     int   = 2000,
    num_samples:    int   = 1000,
    outdir:         str   = "results",
    seed:           int   = 42,
):
    """
    Run NUTS MCMC on a single pixel without vmap, then call diagnose_pixel.

    Uses _estimate_initial_params to initialise NUTS near the correct mode
    rather than relying on a random prior draw, which can land in a wrong
    prior-boundary attractor.
    """
    rng = random.PRNGKey(seed)

    med_flux = float(np.nanmedian(np.abs(spectra[spectra != 0])))
    if a_ha_scale  is None: a_ha_scale  = med_flux
    if a_nii_scale is None: a_nii_scale = med_flux
    if cont_scale  is None:
        cont_scale = float(np.nanpercentile(np.abs(spectra), 90))

    priors = make_priors(
        wave            = wavelength,
        a_ha_scale      = a_ha_scale,
        a_nii_scale     = a_nii_scale,
        continuum_scale = cont_scale,
        v_center        = galaxy_v_kms,
        v_half_range    = v_half_range,
        sigma_inst      = sigma_inst_kms,
    )

    wave_j = jnp.array(wavelength)
    flux_i = jnp.array(spectra[pixel_idx])
    ivar_i = jnp.array(ivar[pixel_idx])

    # Data-driven initialisation — avoids landing in wrong prior-boundary mode
    init_vals = _estimate_initial_params(
        wave           = wavelength,
        flux           = spectra[pixel_idx],
        ivar           = ivar[pixel_idx],
        v_center       = galaxy_v_kms,
        v_half_range   = v_half_range,
        sigma_inst_kms = sigma_inst_kms,
    )
    print(f"\n--- Single-pixel diagnostic: pixel {pixel_idx} "
          f"(x={xy_coords[pixel_idx, 0]}, y={xy_coords[pixel_idx, 1]}) ---")
    print(f"  Data-driven init:  v_los={init_vals['v_los']:.1f} km/s  "
          f"sigma_v={init_vals['sigma_v']:.1f} km/s  "
          f"A_Ha={init_vals['A_Ha']:.3f}  A_NII={init_vals['A_NII']:.3f}")
    print(f"  num_warmup={num_warmup},  num_samples={num_samples}  (no vmap)")

    kernel = NUTS(line_model,
                  init_strategy=init_to_value(values=init_vals))
    mcmc   = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                  num_chains=1, progress_bar=True)
    mcmc.run(rng, wave_j, flux_i, ivar_i, priors)
    raw = mcmc.get_samples()

    single_samples = {k: v[np.newaxis, :] for k, v in raw.items()}

    diagnose_pixel(
        idx     = 0,
        wave    = wavelength,
        spectra = spectra[pixel_idx][np.newaxis, :],
        ivar    = ivar[pixel_idx][np.newaxis, :],
        samples = single_samples,
        outdir  = outdir,
    )

    return single_samples


# ---------------------------------------------------------------------------
# __main__  –  entry point
# ---------------------------------------------------------------------------

def load_data(fname):
    data = SpectralCube.read(fname, hdu=1)
    ivar = SpectralCube.read(fname, hdu=2)
    wav = data.spectral_axis.to(u.AA).value
    return data, ivar, wav

def _ivar_sum(data, axis):
    """Combine ivar slices by summing — correct estimator for spatial binning."""
    return 1 / np.nansum(1/data, axis=axis)

def downsample_cube_spatial(cube, axes, cube_type, factor=2, conv=True):
    cube.allow_huge_operations = True
    for a in axes:
        if cube_type == "flux":
            if conv:
                kernel = Gaussian2DKernel(x_stddev=1)
                cube = cube.spatial_smooth(kernel)
            cube = cube.downsample_axis(factor, a, estimator=np.nansum)
        elif cube_type == "ivar":
            cube = cube.downsample_axis(factor, a, estimator=_ivar_sum)
        else:
            raise ValueError(f"Unknown cube_type: {cube_type!r}")
    return cube

def format_spectra(data_cube, ivar_cube, wav, z, lam_min=6450, lam_max=6700):
    """
    Trim to the fitting window, reshape to (N, M), and apply narrowband S/N mask.
    """
    data = data_cube.unmasked_data[:, :, :].value.copy()
    ivar = ivar_cube.unmasked_data[:, :, :].value.copy()
    wav_mask = (wav >= lam_min*(1+z)) & (wav <= lam_max*(1+z))
    data = data[wav_mask, :, :]
    ivar = ivar[wav_mask, :, :]
    wav = wav[wav_mask]

    n_wave, n_y, n_x = data.shape
    n_bins = n_y * n_x

    spectra      = data.transpose(1, 2, 0).reshape(n_bins, n_wave)
    ivar_spectra = ivar.transpose(1, 2, 0).reshape(n_bins, n_wave)

    nb_mask = (wav >= 6525.6*(1+z)) & (wav <= 6600*(1+z))
    cont = np.median(spectra, axis=1)
    spectra_sub = spectra - cont[:, np.newaxis]
    nb_flux = np.nansum(spectra_sub * nb_mask, axis=1)
    nb_err = np.sqrt(np.nansum(
        (1/np.where(ivar_spectra > 0, ivar_spectra, np.inf)) * nb_mask, axis=1))
    sn = nb_flux / nb_err
    mask = np.isfinite(sn) & (sn > 3)

    yy, xx = np.mgrid[0:n_y, 0:n_x]
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    return spectra_sub[mask], ivar_spectra[mask], xy[mask], wav

# ---------------------------------------------------------------------------
# 11.  Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    spectra:    np.ndarray,
    ivar:       np.ndarray,
    wavelength: np.ndarray,
    xy_coords:  np.ndarray,
    # --- tuning parameters (set these to match your data) ---
    galaxy_v_kms:   float = 5000.0,   # approximate recession velocity [km/s]
    v_half_range:   float = 500.0,    # allowed ± range around galaxy_v_kms
    sigma_inst_kms: float = 30.0,     # instrumental LSF width [km/s]
    flux_scale:     float = None,     # DEPRECATED: use a_ha_scale / a_nii_scale instead
    a_ha_scale:     float = None,     # HalfNormal scale for A_Ha;  None = auto (median |flux|)
    a_nii_scale:    float = None,     # HalfNormal scale for A_NII; None = auto (median |flux|)
    cont_scale:     float = None,     # characteristic continuum level; None = auto
    outdir:         str   = "results",
    seed:           int   = 42,
    run_waic:       bool  = True,     # set False to skip Tier-2 (much faster)
    batch_size:     int   = 512,
    mcmc_cfg:       dict  = None,
):
    """
    Top-level entry point.

    Parameters
    ----------
    spectra       : (N, M) flux array
    ivar          : (N, M) inverse-variance array
    wavelength    : (M,)   wavelength array in Angstroms
    xy_coords     : (N, 2) pixel coordinates
    galaxy_v_kms  : prior centre for recession velocity [km/s]
    v_half_range  : half-width of uniform velocity prior [km/s]
    sigma_inst_kms: instrumental line-spread-function FWHM in km/s
                    (used as hard lower bound on σ_v prior)
    flux_scale    : DEPRECATED — use a_ha_scale / a_nii_scale instead
    a_ha_scale    : HalfNormal scale for A_Ha prior; defaults to median |flux|
    a_nii_scale   : HalfNormal scale for A_NII prior; defaults to median |flux|
    cont_scale    : prior scale for continuum coefficients;
                    defaults to the 90th-percentile flux value
    outdir        : output directory
    seed          : JAX RNG seed
    run_waic      : whether to run Tier-2 WAIC model comparison
    batch_size    : pixels per vmap batch (reduce if GPU OOM)
    mcmc_cfg      : override MCMC_CONFIG defaults
    """
    if mcmc_cfg is None:
        mcmc_cfg = MCMC_CONFIG

    rng = random.PRNGKey(seed)

    # ---- 1.  Sanity checks ------------------------------------------------
    N, M = spectra.shape
    assert ivar.shape       == (N, M), "ivar shape mismatch"
    assert wavelength.shape == (M,),   "wavelength shape mismatch"
    assert xy_coords.shape  == (N, 2), "xy_coords shape mismatch"
    print(f"Fitting {N} spectra with {M} spectral channels each.")

    # ---- 2.  Auto-scale priors --------------------------------------------
    med_flux = float(np.nanmedian(np.abs(spectra[spectra != 0])))
    if a_ha_scale is None:
        a_ha_scale = med_flux
    if a_nii_scale is None:
        a_nii_scale = med_flux
    if cont_scale is None:
        cont_scale = float(np.nanpercentile(np.abs(spectra), 90))

    priors = make_priors(
        wave            = wavelength,
        a_ha_scale      = a_ha_scale,
        a_nii_scale     = a_nii_scale,
        continuum_scale = cont_scale,
        v_center        = galaxy_v_kms,
        v_half_range    = v_half_range,
        sigma_inst      = sigma_inst_kms,
    )
    print(f"Priors:  v_los ~ U({galaxy_v_kms - v_half_range:.0f}, "
          f"{galaxy_v_kms + v_half_range:.0f}) km/s   |   "
          f"σ_v ~ U({sigma_inst_kms:.0f}, 300) km/s   |   "
          f"A_Ha ~ HalfNormal(scale={a_ha_scale:.3e})   |   "
          f"A_NII ~ HalfNormal(scale={a_nii_scale:.3e})")

    # ---- 3.  Run MCMC – line model ----------------------------------------
    print("\n--- Running NUTS MCMC (line model) ---")
    rng, rng_line = random.split(rng)
    line_samples = run_mcmc_all_pixels(
        wavelength, spectra, ivar, priors, rng_line, mcmc_cfg, batch_size
    )

    # ---- 4.  Tier-1 detection: HDI on A_Ha --------------------------------
    print("\n--- Tier-1: HDI detection on A(Hα) ---")
    detected_t1, hdi_lo, hdi_hi = compute_hdi_detection(line_samples["A_Ha"])
    print(f"  Tier-1 detections: {detected_t1.sum()} / {N}")

    # ---- 5.  Convergence diagnostics (R-hat) ------------------------------
    print("\n--- Computing R-hat diagnostics ---")
    rhat = compute_rhat(line_samples)
    n_bad = (rhat["A_Ha"] > 1.05).sum()
    print(f"  Pixels with R̂(A_Ha) > 1.05: {n_bad} / {N}")

    # ---- 6.  Tier-2 detection: WAIC model comparison ----------------------
    if run_waic:
        print("\n--- Running NUTS MCMC (continuum-only null model) ---")
        rng, rng_cont = random.split(rng)
        cont_samples = run_mcmc_continuum(
            wavelength, spectra, ivar, priors, rng_cont, mcmc_cfg, batch_size
        )

        print("\n--- Tier-2: WAIC model comparison ---")
        detected_t2, delta_waic = compute_waic_detection(
            wavelength, spectra, ivar, line_samples, cont_samples, priors, rng
        )
        print(f"  Tier-2 detections: {detected_t2.sum()} / {N}")
        combined = detected_t1 & detected_t2
        print(f"  Combined (T1 ∩ T2): {combined.sum()} / {N}")
    else:
        detected_t2 = detected_t1.copy()
        delta_waic  = np.full(N, np.nan)
        print("  WAIC skipped; using Tier-1 detections only.")

    # ---- 7.  Build 2-D maps -----------------------------------------------
    print("\n--- Building 2-D maps ---")
    maps = build_maps(
        xy_coords, line_samples, detected_t1, detected_t2,
        hdi_lo, hdi_hi, rhat
    )

    # ---- 8.  Save results -------------------------------------------------
    print("\n--- Saving results ---")
    save_results(
        outdir, maps, line_samples, xy_coords,
        detected_t1, detected_t2, hdi_lo, hdi_hi, delta_waic, rhat
    )

    # ---- 9.  Figures -------------------------------------------------------
    print("\n--- Generating figures ---")
    plot_maps(maps, outdir=outdir)
    # Plot a few example fits: brightest, a marginal, and a non-detection
    A_med = np.median(line_samples["A_Ha"], axis=1)
    sorted_idx = np.argsort(A_med)[::-1]
    example_idx = [
        sorted_idx[0],                          # brightest detection
        sorted_idx[len(sorted_idx) // 2],       # median pixel
        sorted_idx[-1],                          # faintest
    ]
    plot_example_fits(wavelength, spectra, ivar, line_samples,
                      indices=example_idx, outdir=outdir)

    print("\nPipeline complete.")
    return maps, line_samples, detected_t1, detected_t2, delta_waic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Emission line fitting pipeline.  "
                    "Run the full image by default, or pass --pixel N to "
                    "run a single-pixel diagnostic without vmap."
    )
    parser.add_argument("fname", help="Path to IFU cube FITS file")
    parser.add_argument("z",     type=float, help="Galaxy redshift")
    parser.add_argument(
        "--pixel", type=int, default=None, metavar="N",
        help="Run single-pixel diagnostic on pixel index N instead of "
             "the full pipeline."
    )
    args = parser.parse_args()

    # ---- shared data loading / preprocessing --------------------------------
    data, ivar, wav = load_data(args.fname)
    data = downsample_cube_spatial(data, [1, 2], "flux", factor=3)
    ivar = downsample_cube_spatial(ivar, [1, 2], "ivar", factor=3)

    spectra, ivar_spectra, xy, wav = format_spectra(data, ivar, wav, args.z)
    wav = wav / (1 + args.z)

    # ---- shared fitting settings (edit here to affect both modes) -----------
    FIT_KWARGS = dict(
        galaxy_v_kms   = 0.0,
        v_half_range   = 400.0,
        sigma_inst_kms = 50.0,
        a_ha_scale     = 100.0,
        a_nii_scale    = 100.0,
        outdir         = "results",
        seed           = 42,
    )

    if args.pixel is not None:
        # ---- single-pixel diagnostic mode -----------------------------------
        run_single_pixel_diagnostic(
            pixel_idx   = args.pixel,
            spectra     = spectra,
            ivar        = ivar_spectra,
            wavelength  = wav,
            xy_coords   = xy,
            num_warmup  = 2000,
            num_samples = 1000,
            **FIT_KWARGS,
        )
    else:
        # ---- full pipeline mode ---------------------------------------------
        maps, samples, det_t1, det_t2, delta_waic = run_pipeline(
            spectra    = spectra,
            ivar       = ivar_spectra,
            wavelength = wav,
            xy_coords  = xy,
            run_waic   = True,
            batch_size = 400,
            mcmc_cfg   = dict(
                num_warmup   = 1000,
                num_samples  = 1000,
                num_chains   = 1,
                progress_bar = True,
            ),
            **FIT_KWARGS,
        )

    N = xy_coords.shape[0]
    rows = []
    for i in range(N):
        rows.append(
            f"{i:6d}  {xy_coords[i,0]:6.1f}  {xy_coords[i,1]:6.1f}  "
            f"{int(detected_t1[i]):3d}  {int(detected_t2[i]):3d}  "
            f"{np.median(samples['v_los'][i]):9.2f}  {np.std(samples['v_los'][i]):9.2f}  "
            f"{np.median(samples['sigma_v'][i]):9.2f}  {np.std(samples['sigma_v'][i]):9.2f}  "
            f"{np.median(samples['A_Ha'][i]):10.4e}  {np.std(samples['A_Ha'][i]):10.4e}  "
            f"{hdi_lo[i]:10.4e}  {hdi_hi[i]:10.4e}  "
            f"{delta_waic[i]:10.3f}  {rhat['v_los'][i]:.4f}  {rhat['A_Ha'][i]:.4f}\n"
        )
    summary_file = os.path.join(outdir, "pixel_summary.txt")
    with open(summary_file, "w") as f:
        f.write(header)
        f.writelines(rows)
    print(f"Saved pixel summary → {summary_file}")

    # --- 2-D FITS maps ---
    hdu_list = [fits.PrimaryHDU()]
    for key, data in maps.items():
        hdu = fits.ImageHDU(data=data.astype(np.float32), name=key.upper())
        hdu_list.append(hdu)
    fits_path = os.path.join(outdir, "emission_maps.fits")
    fits.HDUList(hdu_list).writeto(fits_path, overwrite=True)
    print(f"Saved FITS maps      → {fits_path}")

