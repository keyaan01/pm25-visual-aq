"""Smoke test for C3 abstention (src/abstain.py), CPU only, no network, no model.

  python tests/smoke_abstain.py

Checks: Mahalanobis separates in- vs out-of-distribution features; handcrafted score flags a black /
flat image; thresholds are fit on calibration only; the combined gate checks OOD BEFORE width;
risk-coverage is sensible and AURC/excess-AURC/AUGRC compute; OOD metrics (AUROC/AUPR/FPR95) compute.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import abstain as A  # noqa: E402


def test_mahalanobis_separates():
    rng = np.random.default_rng(0)
    cal = rng.normal(0, 1, size=(300, 32))          # calibration good-image features
    id_test = rng.normal(0, 1, size=(120, 32))      # in-distribution
    ood = rng.normal(3.0, 1, size=(120, 32))        # shifted -> out-of-distribution
    maha = A.fit_mahalanobis(cal)
    s_id = A.mahalanobis_score(maha, id_test)
    s_ood = A.mahalanobis_score(maha, ood)
    assert s_ood.mean() > s_id.mean(), "OOD features must score higher than in-distribution"
    m = A.ood_metrics(s_id, s_ood)
    assert m["AUROC"] > 0.9 and 0 <= m["FPR@95TPR"] <= 1, m
    # round-trip save/load the params
    import tempfile
    d = tempfile.mkdtemp()
    A.save_mahalanobis(d, maha)
    maha2 = A.load_mahalanobis(d)
    assert np.allclose(A.mahalanobis_score(maha2, ood), s_ood)
    print("[abstain] Mahalanobis: AUROC=%.3f FPR@95=%.3f, save/load OK" % (m["AUROC"], m["FPR@95TPR"]))


def test_handcrafted_flags_unusable():
    good = np.random.default_rng(1).uniform(0, 1, size=(64, 64, 3))   # textured, bright
    black = np.zeros((64, 64, 3))                                     # night / unusable
    flat = np.full((64, 64, 3), 0.6)                                  # featureless
    assert A.handcrafted_ood_score(black) > 0.8, "black image must score OOD"
    assert A.handcrafted_ood_score(flat) > 0.8, "flat image must score OOD"
    assert A.handcrafted_ood_score(good) < A.handcrafted_ood_score(black)
    assert A.brightness(black) < 0.05 and A.detail_variance(flat) < 1e-6
    # components + human-readable reason (used by the demo badge)
    dk, fl = A.handcrafted_components(black)
    assert dk > 0.8 and "dark" in A.handcrafted_reason(black)
    assert "flat" in A.handcrafted_reason(flat)
    print("[abstain] handcrafted OOD flags black + flat images (+ named reasons) OK")


def test_thresholds_and_gate_order():
    rng = np.random.default_rng(2)
    # tau_ood = top-k rule: refuse at most `budget` of the good scores (robust to a spike at 0)
    cal_scores = rng.normal(5, 1, size=500)
    tau_ood = A.ood_threshold(cal_scores, false_refusal_budget=0.05)
    rr = (cal_scores > tau_ood).mean()
    assert rr <= 0.05 + 1e-9 and rr >= 0.03, rr                      # ~5% refused, never above budget
    # the real handcrafted case: a distribution spiked at exactly 0 must still respect the budget
    spiked = np.concatenate([np.zeros(950), rng.uniform(0.1, 1.0, 50)])
    tau_sp = A.ood_threshold(spiked, 0.05)
    assert (spiked > tau_sp).mean() <= 0.05 + 1e-9, "zero-spike must not over-refuse"
    cal_widths = rng.uniform(0.5, 2.0, size=500)
    tau_w = A.width_threshold(cal_widths, target_coverage=0.90)
    assert abs(tau_w - np.percentile(cal_widths, 90)) < 1e-9
    # gate order: OOD is checked before width
    assert A.decide(ood_score=99, log_width=0.1, tau_ood=tau_ood, tau_width=tau_w)["answer"] is False
    assert "out-of-distribution" in A.decide(99, 0.1, tau_ood, tau_w)["reason"]
    # a caller-supplied reason (the demo's specific trigger) is passed through
    assert A.decide(99, 0.1, tau_ood, tau_w, ood_reason="too dark (night)")["reason"] == "too dark (night)"
    assert A.decide(0.0, 99, tau_ood, tau_w)["answer"] is False       # uncertainty gate
    assert "uncertain" in A.decide(0.0, 99, tau_ood, tau_w)["reason"]
    assert A.decide(0.0, 0.1, tau_ood, tau_w)["answer"] is True       # confident + in-dist
    print("[abstain] thresholds cal-only (top-k budget, zero-spike safe); gate OOD-before-width OK")


def test_risk_coverage_and_aurc():
    rng = np.random.default_rng(3)
    n = 400
    errors = rng.exponential(1.0, size=n)
    # a GOOD confidence signal correlates with the error; a random one does not
    good_unc = errors + rng.normal(0, 0.2, size=n)
    rand_unc = rng.normal(0, 1, size=n)
    cov, risk = A.risk_coverage_curve(good_unc, errors)
    assert cov[0] < cov[-1] and risk[-1] >= risk[0] - 1e-6            # risk rises with coverage
    assert A.aurc(good_unc, errors) < A.aurc(rand_unc, errors), "good signal must have lower AURC"
    assert A.excess_aurc(good_unc, errors) >= -1e-9
    assert np.isfinite(A.augrc(good_unc, errors))
    # selective metrics at 80% coverage
    y = rng.uniform(0, 300, size=n); point = y + rng.normal(0, 20, size=n)
    iv = np.stack([point - 30, point, point + 30], axis=1)
    m = A.selective_metrics_at_coverage(good_unc, y, point, iv, 0.8)
    assert m["n"] == int(round(0.8 * n)) and {"MAE", "RMSE", "category_accuracy", "interval_coverage"} <= set(m)
    print("[abstain] risk-coverage + AURC (good %.3f < rand %.3f) + selective metrics OK"
          % (A.aurc(good_unc, errors), A.aurc(rand_unc, errors)))


if __name__ == "__main__":
    test_mahalanobis_separates()
    test_handcrafted_flags_unusable()
    test_thresholds_and_gate_order()
    test_risk_coverage_and_aurc()
    print("\nALL ABSTENTION SMOKE CHECKS PASSED")
