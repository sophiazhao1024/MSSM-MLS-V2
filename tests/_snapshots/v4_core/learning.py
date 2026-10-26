"""Scaled independent regressors, bootstrap ensembles and fixed-holdout stopping.

This is an explicitly modern variant. Ensemble spread and classifier scores
are diagnostics/ranking heuristics, not calibrated probability statements.
"""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .domain import BOUNDS, REGRESSION_LABELS, parameter_array, validate_observables


def make_network(settings, seed):
    """Build one scalar network with an isolated, reproducible initialization."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers, width = [], 3
        for _ in range(settings.hidden_layers):
            layers.extend([nn.Linear(width, settings.hidden_width), nn.ReLU()])
            width = settings.hidden_width
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)


def train_with_holdout(model, x, y, xv, yv, settings, seed, *, classifier=False):
    """Restore best validation loss; never use a holdout row in gradients.

The same validation set is reused for stopping, so its loss is not an unbiased
test metric. If no valid holdout rows exist for this task, use fixed epochs and
say so. BCEWithLogitsLoss combines sigmoid with binary cross entropy.
"""
    y, yv = y.reshape(-1, 1), yv.reshape(-1, 1)
    loss_fn = nn.BCEWithLogitsLoss() if classifier else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, stale = float("inf"), None, None, 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), settings.batch_size):
            indices = order[start:start + settings.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite modern training loss")
            loss.backward()
            optimizer.step()
        if len(xv):
            model.eval()
            with torch.inference_mode():
                current = float(loss_fn(model(xv), yv))
            if not np.isfinite(current):
                raise FloatingPointError("Non-finite modern validation loss")
            if current < best_loss:
                best_loss, best_epoch, stale = current, epoch, 0
                best_state = deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= settings.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        loss = float(loss_fn(model(x), y))
    if not np.isfinite(loss):
        raise FloatingPointError("Non-finite final modern training loss")
    return {"epochs_executed": epoch, "best_epoch": best_epoch,
            "training_loss": loss, "training_count_with_replacement": len(x),
            "validation_loss": None if best_state is None else best_loss,
            "validation_count": len(xv),
            "stopping": "holdout_patience" if len(xv) else "fixed_epochs_no_valid_holdout"}


def hb_label(record):
    """Known HB result on a physical row only; missing does not mean excluded."""
    if record.get("status") != "ok":
        return None
    diagnostics = record.get("diagnostics")
    if isinstance(diagnostics, dict) and "hb_result" in diagnostics:
        value = diagnostics["hb_result"]
    else:
        value = record.get("hb_result")
    return value if type(value) is int and value in (0, 1) else None


def _bootstrap_classes(labels, rng):
    """Resample within classes, preserving class counts rather than balancing."""
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])


def _classification_metrics(probability, truth):
    if len(truth) == 0:
        return {"count": 0, "brier": None, "accuracy_at_half": None,
                "positive_recall_at_half": None}
    truth = np.asarray(truth, dtype=bool)
    selected = probability >= .5
    return {"count": len(truth), "brier": float(np.mean((probability - truth) ** 2)),
            "accuracy_at_half": float(np.mean(selected == truth)),
            "positive_recall_at_half": float(np.mean(selected[truth])) if truth.any() else None}


class EnsembleSurrogates:
    """Each ensemble member owns four regressors and separate classifiers."""

    def __init__(self, settings, seed=0):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.settings, self.seed, self.fit_count = settings, seed, 0
        self.members, self.mean, self.scale = [], None, None
        self.hb_enabled = False
        torch.set_num_threads(1)

    @property
    def ready(self):
        return len(self.members) == self.settings.ensemble_size

    @staticmethod
    def inputs(rows):
        if len(rows) == 0:
            return torch.empty((0, 3), dtype=torch.float32)
        x = parameter_array(rows)
        scaled = 2 * (x - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0]) - 1
        return torch.tensor(scaled, dtype=torch.float32)

    @staticmethod
    def transformed_labels(records):
        if not records:
            return np.empty((0, 4), dtype=float)
        values = np.array([validate_observables(r["observables"]) for r in records])
        values[:, 3] = np.log(values[:, 3])
        return values

    def fit(self, records):
        train = [r for r in records if not r.get("validation", False)]
        valid = [r for r in records if r.get("validation", False)]
        physical = [r for r in train if r["status"] == "ok"]
        physical_v = [r for r in valid if r["status"] == "ok"]
        classified = [r for r in train if r["status"] in ("ok", "unphysical")]
        classified_v = [r for r in valid if r["status"] in ("ok", "unphysical")]
        hb_train = [r for r in physical if hb_label(r) is not None]
        hb_valid = [r for r in physical_v if hb_label(r) is not None]
        self.fit_count += 1
        self.members, self.mean, self.scale = [], None, None
        self.hb_enabled = False
        report = {"fit": self.fit_count, "ready": False, "members": [],
                  "physical_train": len(physical), "physical_validation": len(physical_v),
                  "classifier_train": len(classified), "classifier_validation": len(classified_v),
                  "holdout_scope": "reused early-stopping validation, not independent test",
                  "tool_errors_excluded": sum(r["status"] == "tool_error" for r in records)}
        if len(physical) < self.settings.min_physical:
            report["reason"] = "insufficient_physical_training_rows"
            return report
        labels, labels_v = self.transformed_labels(physical), self.transformed_labels(physical_v)
        # Statistics use only valid TRAINING rows, never validation or tool errors.
        self.mean, self.scale = labels.mean(axis=0), labels.std(axis=0)
        self.scale = np.where(self.scale > 1e-12, self.scale, 1.)
        x, xv = self.inputs([r["x"] for r in physical]), self.inputs([r["x"] for r in physical_v])
        y = torch.tensor((labels - self.mean) / self.scale, dtype=torch.float32)
        yv = torch.tensor((labels_v - self.mean) / self.scale, dtype=torch.float32)
        xc = self.inputs([r["x"] for r in classified])
        xcv = self.inputs([r["x"] for r in classified_v])
        class_labels = np.array([r["status"] == "ok" for r in classified], dtype=float)
        class_labels_v = np.array([r["status"] == "ok" for r in classified_v], dtype=float)
        yc, ycv = torch.tensor(class_labels, dtype=torch.float32), torch.tensor(class_labels_v, dtype=torch.float32)
        self.hb_enabled = (len(hb_train) >= self.settings.min_physical
                           and len({hb_label(r) for r in hb_train}) == 2)
        xh, xhv = self.inputs([r["x"] for r in hb_train]), self.inputs([r["x"] for r in hb_valid])
        hb_labels = np.array([hb_label(r) for r in hb_train], dtype=float)
        yhb = torch.tensor(hb_labels, dtype=torch.float32)
        yhvb = torch.tensor([hb_label(r) for r in hb_valid], dtype=torch.float32)
        for member_index in range(self.settings.ensemble_size):
            base_seed = self.seed + self.fit_count * 1000 + member_index * 100
            member_report = {"member": member_index, "regressors": []}
            regressors = []
            for j, label in enumerate(REGRESSION_LABELS):
                seed = base_seed + j
                rng = np.random.default_rng(seed)
                indices = rng.integers(0, len(x), size=len(x))
                model = make_network(self.settings, seed)
                stats = train_with_holdout(model, x[indices], y[indices, j], xv, yv[:, j],
                                           self.settings, seed)
                stats.update(label=label, seed=seed, bootstrap_unique=int(len(np.unique(indices))))
                member_report["regressors"].append(stats)
                regressors.append(model)
            seed = base_seed + 90
            indices = _bootstrap_classes(class_labels, np.random.default_rng(seed))
            classifier = make_network(self.settings, seed)
            member_report["physical_classifier"] = train_with_holdout(
                classifier, xc[indices], yc[indices], xcv, ycv, self.settings, seed, classifier=True)
            hb_classifier = None
            if self.hb_enabled:
                seed = base_seed + 91
                indices = _bootstrap_classes(hb_labels, np.random.default_rng(seed))
                hb_classifier = make_network(self.settings, seed)
                member_report["hb_classifier"] = train_with_holdout(
                    hb_classifier, xh[indices], yhb[indices], xhv, yhvb,
                    self.settings, seed, classifier=True)
            self.members.append((regressors, classifier, hb_classifier))
            report["members"].append(member_report)
        report.update(ready=True, input_scaling="known bounds to [-1,1]",
                      output_scaling="train-only zscore after natural-log omega",
                      output_mean=self.mean.tolist(), output_scale=self.scale.tolist(),
                      hb_classifier_enabled=self.hb_enabled,
                      hb_training_count=len(hb_train), hb_validation_count=len(hb_valid),
                      hb_disabled_reason=None if self.hb_enabled else "insufficient_rows_or_only_one_known_class")
        predicted = self.predict([r["x"] for r in physical])
        predicted_v = self.predict([r["x"] for r in physical_v])
        report["regression_metrics"] = []
        for j, name in enumerate(REGRESSION_LABELS):
            report["regression_metrics"].append({
                "label": name, "space": "original value; natural log for omega",
                "training_mse": float(np.mean((predicted["mean_transformed"][:, j] - labels[:, j]) ** 2)),
                "validation_mse": float(np.mean((predicted_v["mean_transformed"][:, j] - labels_v[:, j]) ** 2)) if len(labels_v) else None,
                "validation_count": len(labels_v)})
        report["physical_validation_metrics"] = _classification_metrics(
            self.predict([r["x"] for r in classified_v])["physical_probability"], class_labels_v)
        report["hb_validation_metrics"] = (_classification_metrics(
            self.predict([r["x"] for r in hb_valid])["hb_probability"],
            [hb_label(r) for r in hb_valid]) if self.hb_enabled else {"count": len(hb_valid), "model_enabled": False})
        return report

    def predict(self, x):
        """Return all predictions; classifier scores never suppress a row.

omega = exp(mean(ln omega)) is a geometric estimate, not E[omega]. Ensemble
standard deviations are measured in standardized transformed-label space and
have no coverage guarantee. No physical clipping changes target acceptance.
"""
        if not self.ready:
            raise RuntimeError("Ensemble is not fitted")
        tensor = self.inputs(x)
        with torch.inference_mode():
            outputs = np.stack([np.column_stack([m(tensor).squeeze(1).numpy() for m in regs])
                                for regs, _, _ in self.members])
            probabilities = np.stack([torch.sigmoid(c(tensor)).squeeze(1).numpy()
                                      for _, c, _ in self.members])
            hb_probabilities = (np.stack([torch.sigmoid(h(tensor)).squeeze(1).numpy()
                                         for _, _, h in self.members])
                                if self.hb_enabled else np.ones((len(self.members), len(tensor))))
        transformed = outputs.mean(axis=0).astype(float) * self.scale + self.mean
        values = transformed.copy()
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values[:, 3] = np.exp(values[:, 3])
        spread = outputs.std(axis=0).astype(float)
        return {"mean": values, "mean_transformed": transformed,
                "std_standardized": spread,
                "uncertainty": np.sqrt(np.mean(spread ** 2, axis=1)),
                "physical_probability": probabilities.mean(axis=0).astype(float),
                "hb_probability": hb_probabilities.mean(axis=0).astype(float)}

    def export(self):
        """Only tensors and basic Python values, compatible with weights_only=True."""
        return {"kind": "mssm_mls_modern_ensemble_v1", "settings": asdict(self.settings),
                "seed": self.seed, "fit_count": self.fit_count,
                "mean": None if self.mean is None else self.mean.tolist(),
                "scale": None if self.scale is None else self.scale.tolist(),
                "hb_enabled": self.hb_enabled,
                "members": [{"regressors": [m.state_dict() for m in regs],
                             "classifier": classifier.state_dict(),
                             "hb_classifier": None if hb is None else hb.state_dict()}
                            for regs, classifier, hb in self.members]}

    def load_export(self, state):
        """Load this exact architecture/settings; the scan runner owns RNG resume."""
        if state.get("kind") != "mssm_mls_modern_ensemble_v1" or state.get("settings") != asdict(self.settings):
            raise ValueError("Modern model state has incompatible kind or settings")
        if state.get("seed") != self.seed:
            raise ValueError("Modern model seed differs")
        self.fit_count = int(state["fit_count"])
        self.hb_enabled = bool(state["hb_enabled"])
        self.mean = None if state["mean"] is None else np.asarray(state["mean"], dtype=float)
        self.scale = None if state["scale"] is None else np.asarray(state["scale"], dtype=float)
        self.members = []
        for member in state["members"]:
            if len(member["regressors"]) != 4:
                raise ValueError("Expected four independent regressors")
            regressors = [make_network(self.settings, 0) for _ in range(4)]
            for model, weights in zip(regressors, member["regressors"]):
                model.load_state_dict(weights); model.eval()
            classifier = make_network(self.settings, 0)
            classifier.load_state_dict(member["classifier"]); classifier.eval()
            hb = None
            if self.hb_enabled:
                hb = make_network(self.settings, 0)
                hb.load_state_dict(member["hb_classifier"]); hb.eval()
            self.members.append((regressors, classifier, hb))
        if self.members and (not self.ready or self.mean.shape != (4,) or self.scale.shape != (4,)
                             or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                             or np.any(self.scale <= 0)):
            raise ValueError("Invalid ensemble normalization state")
        return self
