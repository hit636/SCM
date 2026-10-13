# Methodology

## Evidence and budgets

A competence cell is one entity–expert pair. Its label is the mean of the expert's
per-origin RMSE over admissible mining origins. A forecast request is a separate
execution unit identified by stage, entity, expert, origin, expert configuration
hash, and protocol hash. Only identical request keys reuse forecasts.

The mining budget is `floor(rho * executable_cells)`. Initialization acquires the
LightGBM anchor on every row and one eligible random row for each other expert.
Every initialized cell is charged. Four batches divide the remainder with sizes
differing by at most one cell. Acquisition scores are frozen within a batch.

The learner receives acquired losses and masks. Hidden labels enter only the
post-selection research audit. Router fitting reads forecasts already acquired
at mining origins and creates no expert requests.

## Forecastability fingerprints

The temporal block contains lifecycle duration, valid steps, missing fraction,
zero ratio, ADI, positive-value CV squared, mean, standard deviation, coefficient
of variation, quantiles, maximum, consecutive-valid-change burstiness, outlier
ratio, winsorized least-squares slope, trend/seasonal strengths, autocorrelations, spectral entropy,
band energies, recent means, and rolling variance. Numeric context includes
missing flags and historical response statistics. Categorical vocabularies and
feature standardization fit on training history only.

Invalid grid positions use frozen per-entity training means for decomposition
and Fourier features. Unseen entities use the training-cohort mean of those
fill values. Observed-day statistics use the original validity mask. Full FFT
power includes the DC component; band boundaries use absolute frequency and
month/week periods in the declared time-step units.

## Competence recovery

The normalized prediction follows Eq. (6):

\[
\hat\ell^n_{i,m}=\mu+b_i+d_m+h_\theta(z_i)^T v_m.
\]

The encoder has two 128-unit ReLU layers followed by a 32-dimensional linear
output. Row biases use training-row identities; unseen rows use zero bias.
Each expert owns a factor and scalar bias. Width 32 does not impose a strict
low-rank constraint when the expert pool contains fewer than 32 entries.

The objective combines acquired-entry Huber regression, observed-pair
softplus ranking, expert-factor regularization, and encoder regularization.
The ranking mask uses `measured_a + rank_margin < measured_b`. Hidden entries
never form reconstruction labels or ranking pairs.

Seed losses determine a frozen row median and MAD. Degenerate MAD uses a
training-only scale proxy: the stratum median of consecutive-valid first-
difference standard deviations. Strata cross at least 90 valid training steps
with zero ratio above 0.5; a positive configured floor handles degenerate
scales. This is the repository's explicit fallback statistic.

Five independently initialized and masked models supply original-unit mean
loss and standard deviation. Training uses Adam, gradient clipping, and early
stopping on the acquired-label training objective.

## Acquisition

Eq. (12) is implemented as

\[
A_{i,m}=s_{i,m}+\alpha\max(0,\ell_i^{best}-\hat\ell_{i,m})
+\beta(n_m+1)^{-1/2}-\zeta\tilde c_m.
\]

Loss predictions, uncertainty, and improvement use restored loss units.
Offline costs are normalized to the anchor. Observed or unavailable cells receive
negative infinity. Stable row-major ordering resolves equal scores.

Single-column insertion uses the same score equation with the appended column's
remaining eligible rows. Constant coverage/cost terms for a single column do
not change within-column ordering.

## Teacher and router

Observed cells retain measured losses; unobserved cells use predicted loss plus
`eta * uncertainty`. Teacher logits are
`-(adjusted_loss + nu * online_cost) / tau`. Availability masking precedes softmax.

The router has two 128-unit ReLU layers with dropout after each hidden ReLU and
one linear output head per expert. The objective sums cached-forecast MSE,
teacher KL, squared mean-probability balance, and expected online cost. The
prediction term renormalizes probabilities over observed experts only.

The gate applies availability, renormalizes over available experts, selects
Top-r, and prunes probabilities below the threshold. If pruning removes every
expert, the highest-probability available expert is retained. Fusion weights
are renormalized over the final retained set.

Teacher parameters are fixed in the YAML. The calibration stage selects a
threshold from `router.calibration_thresholds` using the configured primary
metric; ties favor fewer executions, then the lower threshold.

## Costs and metrics

Training-only scalar requests measure expert costs, including local statistical
fitting. Online costs are relative to the anchor. Offline cost weights multiply
per-request medians by the number of mining origins. Weighted mining cost is
acquired weighted work divided by full executable weighted work.

Elapsed mining, router, and calibration times form selected construction time.
Shared backbone training, data/features, and profiling are recorded separately.
Test execution and hidden-label audits are evaluation work. Summaries report
stage-specific cell counts and distinct mining/calibration entity–expert pairs.

Pooled RMSE, MSE, MAE, WAPE, and sMAPE use common valid targets and weights.
WAPE and sMAPE are ratios. Hourly evaluations use configured train-zscore units;
retail/daily evaluations use raw sales units. Neural-seed metrics are averaged
within each evidence mask before mask-level means and standard deviations.

## Reuse

Identity holdout freezes competence/router training and feature normalization
without excluding legal histories from every global expert backbone. New-column
updates train only appended competence factors/biases and router-head parameters.
Old logits remain unchanged; softmax probabilities and executed mixtures can
change after insertion.

The soft-teacher theorem in the manuscript concerns weighted component-expert
loss. This code evaluates fused forecasts directly; it does not calculate a
sample-complexity guarantee or an empirical theorem upper bound.

## Revision details

The competence minibatch reconstruction numerator is divided by
`global_observed_count * batch_rows / N`; ranking is divided analogously by the
global eligible pair count. This estimates the declared global mean objective
under uniform row batching rather than equally weighting locally sparse batches.
Pairs are counted in row chunks to avoid allocating an N×M×M tensor.
`reconstruction: mse` and `row_normalization: false` expose their ablations.
The empirical margin uses acquired seed-origin histories only and freezes before
active rounds. Last-positive recency and recent moving-mean change are appended
features; existing metadata feature indices remain stable.

NN-EI/c uses the same shared neural Huber ensemble as SCM with lambda_rank=0.
Its acquisition score is Gaussian expected improvement divided by positive
anchor-normalized offline cost. Zero uncertainty uses the deterministic positive
improvement limit. PMF-A retains its separately specified GP posterior control.
Independent neural recovery fits a separate two-hidden-layer model per expert
and member on acquired labels only.

Straight-through/Gumbel controls select at most r substantial weights, renormalize
on that support, and use a soft surrogate gradient during fitting. Gumbel noise
is used only in training; evaluation is deterministic. Sparsemax uses its sorted
support threshold projection. An epsilon probability floor permits finite KL;
it is recorded in implementation and is not mathematically exact zero support.
