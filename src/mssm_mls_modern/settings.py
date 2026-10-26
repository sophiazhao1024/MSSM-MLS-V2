"""Explicit engineering choices for the modern experiment, not paper settings."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModernSettings:
    budget: int = 20000
    initial: int = 1000
    exploit: int = 63
    uncertain: int = 21
    random_extra: int = 21
    ensemble_size: int = 3
    surrogate_kind: str = "gp"
    gp_steps: int = 100
    gp_max_anchors: int = 256
    gp_geometry: str = "mh2_signed_detuning_hbratio_v3"
    gp_omega_input_dim: int = 4
    gp_hb_input_dim: int = 4
    pool_size: int = 16384
    epochs: int = 300
    patience: int = 30
    learning_rate: float = 0.001
    batch_size: int = 128
    hidden_width: int = 64
    hidden_layers: int = 2
    validation_fraction: float = 0.2
    min_physical: int = 12
    diversity_radius: float = 0.05
    # Numerical log guard only; random slots, not this floor, provide exploration.
    probability_floor: float = 1e-12
    local_fraction: float = 0.5
    local_elites: int = 8
    local_scale_small: float = 0.01
    local_scale_medium: float = 0.03
    local_scale_large: float = 0.1
    mh_soft_width: float = 5.0
    mH_soft_width: float = 1.0
    chi2_soft_width: float = 10.0
    lnomega_soft_width: float = 0.3
    max_consecutive_tool_errors: int = 5
    target_policy: str = "paper-higgs"

    def __post_init__(self):
        positive_counts = ("budget", "initial", "random_extra", "ensemble_size",
                           "pool_size", "epochs", "patience", "batch_size",
                           "hidden_width", "hidden_layers", "min_physical",
                           "max_consecutive_tool_errors", "local_elites", "gp_steps", "gp_max_anchors")
        for key in positive_counts:
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("exploit", "uncertain"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        if self.exploit + self.uncertain == 0:
            raise ValueError("At least one model-guided selection is required")
        if self.initial < 5 or self.budget < self.initial:
            raise ValueError("Require budget >= initial >= 5")
        if self.ensemble_size < 2 or self.min_physical < 2:
            raise ValueError("Need at least two ensemble members and physical rows")
        if self.gp_max_anchors < 4:
            raise ValueError("gp_max_anchors must be at least four")
        if self.surrogate_kind not in ("gp", "ensemble"):
            raise ValueError("Unknown surrogate_kind")
        if (self.gp_geometry != "mh2_signed_detuning_hbratio_v3"
                or type(self.gp_omega_input_dim) is not int or self.gp_omega_input_dim != 4
                or type(self.gp_hb_input_dim) is not int or self.gp_hb_input_dim != 4):
            raise ValueError("This variant requires the explicit resonance and HB mass feature geometry")
        if self.pool_size < self.exploit + self.uncertain:
            raise ValueError("pool_size is too small for model-guided selections")
        for key in ("learning_rate", "mh_soft_width", "mH_soft_width",
                    "chi2_soft_width", "lnomega_soft_width"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.diversity_radius) or self.diversity_radius < 0:
            raise ValueError("diversity_radius must be finite and non-negative")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if int(self.initial * self.validation_fraction) < 1:
            raise ValueError("Initial holdout must contain at least one row")
        if not 0 < self.probability_floor <= 1e-6:
            raise ValueError("probability_floor is a numerical guard, require (0, 1e-6]")
        if not 0 <= self.local_fraction < 1:
            raise ValueError("local_fraction must be in [0, 1), preserving global proposals")
        if any(isinstance(v, bool) or not math.isfinite(v) or not 0 < v <= 1 for v in self.local_scales):
            raise ValueError("local_scales must contain finite positive unit-box scales <= 1")
        if self.target_policy not in ("paper-higgs", "eq7-only"):
            raise ValueError("Unknown target_policy")

    @property
    def batch_points(self):
        return self.exploit + self.uncertain + self.random_extra

    @property
    def local_scales(self):
        return (self.local_scale_small, self.local_scale_medium, self.local_scale_large)

