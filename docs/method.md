# Published method, V5 update and interpretation

## Physics and target

The scanned parameters are x=(M1, mu, tan_beta), with M1 in [2,20] GeV, mu in [7000,9000] GeV, and tan_beta in [2,10]. Fixed parameters and input cards are in `domain.py` and `cards.py`. The heavy CP-even H is the observed Higgs candidate. Exact Eq.(7) hard cuts used here are:

```text
123 < mH < 129 GeV
chi2_HS < 112.7273
0.08209 < Omega*h^2 < 0.15511
L = indicator(these cuts) * exp(-0.15*mh)
```

The main `paper-higgs` target additionally requires reliable true HiggsBounds result exactly integer1. The paper's Eq.(7) does not explicitly display this HB indicator; its text requires consistency with Higgs data. The original HB integration/card settings were not published, so this remains a declared reconstruction convention. `eq7-only` is a separate diagnostic, never silently substituted for the main metric. Exact cut endpoints are excluded. Technical failures remain failures in the logical denominator and supply no invented regression/physical-class labels.

## What the article actually disclosed

Ren et al., Nuclear Physics B943 (2019)114613, DOI10.1016/j.nuclphysb.2019.114613: Model4 on PDF pp.7–10; Eq.(7) on p.9; Fig.8 and training continuation on p.10. The article uses four separate regressors for mh, mH, total HiggsSignals chi2 and ln(Omega*h^2), plus a physical-validity classifier. Each has four hidden layers of60 units. Adam trains for **up to2000 epochs**. It describes1000 initial random points,100 guided plus5 additional random points per iteration, cumulative training data and rejection sampling from the predicted likelihood. The MLS text explicitly describes no independent validation/test/regularization in that original loop. Offline validation in the toy-model comparison is a different experiment.

The article did not disclose all learning rates, initialization/retraining state, exact early stopping, random seeds, proposal-cap/RNG details, author input cards or complete numerical settings. Fixed2000 epochs, fresh weights every iteration and standardization in the strong benchmark are reconstruction choices. The raw-input/log-omega `historical` path is a separate reconstruction and may stall; it is not the stronger comparator used for the published seed0 table. xBIT is a later implementation reference and differs in its network/training/selection details; it cannot recover missing original source.

## Strong reference versus V5

`strong-reference` is the stronger standardized four-NN reconstruction, with the original-shaped 4x60 networks, classifier and rejection sampling; default epochs2000,100+5 batch and200-million finite cheap-proposal cap. It uses mean/std output scaling and box-scaled inputs. The original article does not prove those scaling choices were used by the authors.

V5's configured modern path (`surrogate_kind=gp`) instead fits four scalar GPs for mh^2, mH, chi2 and lnOmega, plus a fifth scalar GP for log positive measured HB exclusion/limit ratio when available. Classifiers model physical validity, observed solver success, and binary HB fallback. Missing-class support can use explicit fallback constants, so it is not correct to say all five GPs and every classifier are always trained for every dataset.

All parameter inputs are mapped by known box bounds; training labels and added resonance/HB features are scaled using training data. The relic-density input includes signed detuning based on predicted mh, with approximate propagation of its GP spread. Hyperparameters/anchors are bounded by the frozen configuration; this is not an exact full-dataset GP posterior. Conditional latent GP spread and the marginal feasibility product are working approximations, not calibrated joint probabilities or posterior samples.

V5 includes a numerically stable exact log probability for positive mh^2, preserving small tails instead of squaring a previous probability factor. Exploitation ranks an approximate joint feasibility product; exploration uses log Bernoulli entropy of that same uncalibrated product, not Bayesian mutual information. Default batches have63 exploit,21 entropy and21 unconditional random points, interleaved so exploration occurs early. True physical/solver/HB labels are kept distinct. Local/global candidates, source-truth elite diversity and soft distance repulsion avoid treating repeated nearby predictions as coverage evidence. **The V6 local covariance geometry is not included.**

An initial uniform subset is held out for internal early stopping; it is not an independent final test set. Seed0 was repeatedly used during development, so development selection can overfit this one initialization/physics profile. New paired seeds remain necessary for a general-performance claim.

## What the data show

At20000 attempts, main targets rise4248→4642 (+394,+9.27495% relative); Eq7-only falls10974→6728. Thus improvement is specifically under the joint HB acceptance target and does not mean all physical criteria improved. Grid20/40 occupation equals the strong reference, grid100 occupation is448 vs398; greedy spacing-filtered counts are3332 vs3291 at a prespecified normalized radius1e-3. These diagnostics do not prove that all allowed islands were found.

The old >=10%/4673-target development condition failed. This is a completed V5 study with a recorded failure of that prespecified condition; the protocol result is unchanged. There is no completed blind multi-seed replication and no reproduction of the original paper's ~80% reported cumulative efficiency. That paper compared MLS to MultiNest; the uniform-random experiment here is an additional comparator.

Physics settings reconstructed from available descriptions/official examples include FHflags4002420111, topmass172GeV, HSdataset latestresults-1.4.0, HSrunmode1/pdf2, mass uncertainties and MO fast1/Beps1e-5/VW=VZ=0. Author cards/options are unavailable. These choices are recorded, not claimed as recovered originals; source-version agreement alone does not establish identical numerical physics.
