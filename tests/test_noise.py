import numpy as np
from p1.noise import flip_labels, validation_label_views


def test_fixed_rng_reproducibility_and_changes():
    original = np.arange(100) % 10
    a, changed = flip_labels(original, 0.4, 10, np.random.default_rng(12))
    b, changed_b = flip_labels(original, 0.4, 10, np.random.default_rng(12))
    assert np.array_equal(a,b)
    assert np.array_equal(changed, changed_b)
    assert np.all(a[changed] != original[changed])
    assert np.array_equal(original, np.arange(100)%10)


def test_weighted_uniform_matches_hetero_in_expectation():
    labels = np.arange(160) % 10
    clients = {0:list(range(0,80)), 1:list(range(80,160))}
    views, metadata = validation_label_views(labels, clients, [0.0, 0.6], seed=77)
    assert metadata['uniform']['0']['planned_rate'] == 0.3
    assert metadata['uniform']['1']['planned_rate'] == 0.3
    assert np.array_equal(views['clean'][0], labels[clients[0]])


def test_uniform_symmetric_noise_ranking_expected_math():
    c, eta = 10, 0.4
    slope = 1 - (c * eta)/(c - 1)
    assert slope > 0
    def observed(acc): return slope*acc + eta/(c - 1)
    assert observed(0.8) > observed(0.7)


def test_expected_accuracy_matches_explicit_transition_matrix():
    # Independent enumeration, including the ranking boundary and reversal.
    for c in (2, 10):
        truth = np.arange(100) % c
        pred = truth.copy()
        pred[:30] = (pred[:30] + 1) % c
        clean_acc = np.mean(pred == truth)
        for eta in (0., .2, (c - 1) / c, 1.):
            transition = np.full((c, c), eta / (c - 1))
            np.fill_diagonal(transition, 1 - eta)
            expected = np.mean(transition[truth, pred])
            formula = (1 - c * eta / (c - 1)) * clean_acc + eta / (c - 1)
            assert np.isclose(expected, formula)
