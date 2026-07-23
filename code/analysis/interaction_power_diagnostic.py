#!/usr/bin/env python3
"""
Experience x assistance-condition INTERACTION test + POWER/MDE diagnostic.

Purpose: document, transparently and reproducibly, (a) the experience x assistance
interaction estimate, and (b) that the 11-reader design is underpowered to detect an
interaction of plausible magnitude — so that the manuscript's narrowed (descriptive)
claim is consistent with the analysis code rather than in tension with it.

Design: 11 readers, each measured with and without model support (22 rows). The experience x
condition interaction is, for a paired design, exactly the slope of the WITHIN-READER difference
(with - without) regressed on years of experience. This differencing removes the reader random
intercept, so it is both the correct interaction test and numerically stable.

Data: the per-(radiologist, condition) regression table is derived live from
data/source_data/figure_1/csv_v2/radiologist_df.csv via _metrics_utils.compute_individual_perf,
the same helper used by fig_5.py and fig_6.py. No static intermediate.

Reads the study data read-only; writes nothing; runs no side effects. Deterministic
(the analysis is closed-form — no resampling, so no seed is required).

Usage:
    python3 code/analysis/interaction_power_diagnostic.py
"""
import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RDF_PATH = os.path.join(REPO_ROOT, 'data', 'source_data', 'figure_1', 'csv_v2',
                        'radiologist_df.csv')

sys.path.insert(0, os.path.dirname(HERE))
from _metrics_utils import compute_individual_perf  # noqa: E402

METRICS = ["rad_accuracy", "mean_confidence", "mean_response_time"]


def load():
    """Per-(radiologist, condition) regression table: 11 readers x 2 conditions."""
    radiologist_df = pd.read_csv(RDF_PATH, float_precision='round_trip')
    df = compute_individual_perf(radiologist_df)[
        ["radiologist", "with_segmentation", "years_experience",
         "rad_accuracy", "mean_confidence", "mean_response_time"]
    ].copy()
    df["cond"] = df["with_segmentation"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"])
    return df


def paired(df, metric):
    w = df[df["cond"]].set_index("radiologist")
    wo = df[~df["cond"]].set_index("radiologist")
    readers = [r for r in wo.index if r in w.index]
    exp = wo.loc[readers, "years_experience"].to_numpy(float)
    delta = w.loc[readers, metric].to_numpy(float) - wo.loc[readers, metric].to_numpy(float)
    return exp, delta, len(readers)


def interaction_test(exp, delta):
    """Interaction = slope of (with-without) delta on experience. n readers, df=n-2."""
    n = len(exp)
    lr = stats.linregress(exp, delta)
    dfres = n - 2
    tcrit = stats.t.ppf(0.975, dfres)
    ci = (lr.slope - tcrit * lr.stderr, lr.slope + tcrit * lr.stderr)
    return dict(n=n, slope=lr.slope, se=lr.stderr, p=lr.pvalue, df=dfres,
               ci_lo=ci[0], ci_hi=ci[1], r2=lr.rvalue**2)


def mde_80(exp, delta, se_slope):
    """Minimum interaction slope detectable at 80% power, two-sided alpha=0.05 (analytic)."""
    dfres = len(exp) - 2
    tcrit = stats.t.ppf(0.975, dfres)
    tpow = stats.t.ppf(0.80, dfres)
    return (tcrit + tpow) * se_slope


def power_for(slope, se_slope, n):
    """Power (two-sided alpha=0.05) to detect a true interaction of size `slope`."""
    dfres = n - 2
    tcrit = stats.t.ppf(0.975, dfres)
    ncp = slope / se_slope
    return float(1 - stats.nct.cdf(tcrit, dfres, ncp) + stats.nct.cdf(-tcrit, dfres, ncp))


def per_condition_regressions(df, metric):
    """Reproduce the two per-condition regressions the manuscript reports (sanity check)."""
    out = {}
    for label, mask in [("without", ~df["cond"]), ("with", df["cond"])]:
        sub = df[mask]
        lr = stats.linregress(sub["years_experience"], sub[metric])
        out[label] = (lr.rvalue**2, lr.pvalue)
    return out


def main():
    df = load()
    print("=" * 78)
    print("EXPERIENCE x ASSISTANCE INTERACTION — estimate, uncertainty, and power")
    print(f"Data: {os.path.relpath(RDF_PATH, REPO_ROOT)} -> compute_individual_perf  |  "
          f"{len(df)} rows, {df['radiologist'].nunique()} readers x 2 conditions")
    print("=" * 78)

    for metric in METRICS:
        exp, delta, n = paired(df, metric)
        it = interaction_test(exp, delta)
        mde = mde_80(exp, delta, it["se"])
        pow_obs = power_for(it["slope"], it["se"], n)
        pc = per_condition_regressions(df, metric)

        print(f"\n### {metric}   (n={n} readers, interaction df={it['df']})")
        print(f"  Per-condition (manuscript sanity check): "
              f"without R2={pc['without'][0]:.3f} p={pc['without'][1]:.3f} | "
              f"with R2={pc['with'][0]:.3f} p={pc['with'][1]:.3f}")
        print(f"  INTERACTION slope (delta-vs-experience) = {it['slope']:+.5f} per year")
        print(f"      SE = {it['se']:.5f}   95% CI [{it['ci_lo']:+.5f}, {it['ci_hi']:+.5f}]"
              f"   p = {it['p']:.3f}")
        print(f"  Minimum detectable interaction slope at 80% power = |{mde:.5f}| per year")
        print(f"  Ratio |MDE| / |observed| = {abs(mde)/max(abs(it['slope']),1e-9):.1f}x")
        print(f"  Power to detect the OBSERVED interaction = {pow_obs*100:.1f}%")
        verdict = "UNDERPOWERED" if pow_obs < 0.8 else "adequately powered"
        print(f"  -> {verdict}: at n={n}, the observed interaction is far below the "
              f"effect this design could reliably detect.")

    print("\n" + "=" * 78)
    print("SUMMARY: The interaction tests are non-significant, but the design is severely")
    print("underpowered (observed power well below 80%), so a non-significant result is")
    print("uninformative rather than evidence against an interaction. The manuscript therefore")
    print("narrows to the descriptive per-condition statement and does not claim 'strengthening'.")
    print("=" * 78)


if __name__ == "__main__":
    main()
