"""Reproducibility of `simulate_copula` across processes when hub reuse is enabled.

`_set_corr` picks a hub out of the `hub_eligible` **set**. Iterating a set of strings follows hash
order, and Python randomises string hashing per process (PEP 456), so without an explicit ordering
`random.choice` selects a different hub in every process even from identical RNG state -- seeding
both `random` and `np.random` is not enough. That silently changes which pairs get planted, and
therefore the simulated data, between otherwise-identical runs.

These tests spawn real subprocesses with different PYTHONHASHSEED values, because the effect cannot
be reproduced in-process: the hash seed is fixed for the lifetime of the interpreter.
"""

import os
import subprocess
import sys

import pytest

# hub_reuse_prob=1.0 with several same-type edges guarantees the reuse branch is taken and that
# `hub_eligible` holds more than one candidate, which is exactly when iteration order matters.
# A weak background keeps the hub ceiling (lambda_min**2) loose enough that reuse is accepted.
SNIPPET = """
import random
import numpy as np
import modina

random.seed(0)
np.random.seed(0)
context1, _, _, ground_truth, _ = modina.simulate_copula(
    n_cont=20, n_bi=0, n_cat=0, n_samples_1=200, n_samples_2=200,
    n_corr_cont_cont=5, hub_reuse_prob=1.0,
    corr_low=0.2, corr_high=0.3, shift_low=0.2, shift_high=0.3,
    background_model='scale_free', background_strength=0.1, background_seed=0)
print("PAIRS:" + "|".join("_".join(sorted(p)) for p in ground_truth[1]))
print("CHECKSUM:%.10f" % float(abs(context1.to_numpy()).sum()))
"""


def _run_with_hashseed(hashseed):
    """Run the snippet in a fresh interpreter under a specific PYTHONHASHSEED."""
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    done = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True, text=True,
                          env=env, timeout=300)
    assert done.returncode == 0, f"subprocess failed:\n{done.stderr}"
    pairs = next(l[len("PAIRS:"):] for l in done.stdout.splitlines() if l.startswith("PAIRS:"))
    checksum = next(l[len("CHECKSUM:"):] for l in done.stdout.splitlines()
                    if l.startswith("CHECKSUM:"))
    return pairs, checksum


@pytest.fixture(scope="module")
def runs():
    return [_run_with_hashseed(seed) for seed in (0, 1, 12345)]


def test_hub_reuse_actually_fires(runs):
    """Guard: if reuse never happened, the ordering path is untouched and the tests below would
    pass trivially even with the bug present."""
    pairs = runs[0][0].split("|")
    nodes = [n for pair in pairs for n in pair.split("_")]
    assert len(nodes) - len(set(nodes)) >= 1, (
        f"no node is shared between planted pairs, so hub reuse never fired: {pairs}")


def test_planted_pairs_identical_across_hash_seeds(runs):
    """Same seed => same planted pairs, whatever PYTHONHASHSEED the process happens to get."""
    pairs = {r[0] for r in runs}
    assert len(pairs) == 1, (
        "simulate_copula planted different pairs under different PYTHONHASHSEED values, so it is "
        f"not reproducible across processes: {pairs}")


def test_simulated_data_identical_across_hash_seeds(runs):
    """The consequence that actually matters: the data itself must not depend on the hash seed."""
    checksums = {r[1] for r in runs}
    assert len(checksums) == 1, (
        f"simulate_copula produced different data under different PYTHONHASHSEED values: {checksums}")
