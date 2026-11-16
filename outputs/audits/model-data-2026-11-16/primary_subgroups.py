"""Reproduce exploratory subgroup checks from the frozen five-seed replay."""
import json

import numpy as np

from audit import REPO_ROOT, save
from predict_badminton.tournament_catalog import TournamentCatalog


def main():
    source = REPO_ROOT / "outputs/experiments/country-vs-production-promotion-2026-11-16/paired-clusters-rolling_2026.json"
    data = json.loads(source.read_text())
    catalog = TournamentCatalog.load()
    world_tour = []
    for cluster in data["clusters"]:
        edition = catalog.lookup(cluster.rsplit(":", 1)[0])
        world_tour.append(bool(edition and "world tour" in edition["category"].casefold()))
    counts = np.asarray(data["match_counts"])
    sums = np.asarray(data["paired_metric_sums_by_seed_cluster"])
    summary = {}
    for name, mask in (("world_tour", np.asarray(world_tour)), ("other", ~np.asarray(world_tour))):
        per_seed = sums[:, mask].sum(axis=1) / counts[mask].sum()
        summary[name] = {"n": int(counts[mask].sum()), "clusters": int(mask.sum()),
                         "mean_deltas": per_seed.mean(axis=0).tolist(), "per_seed_deltas": per_seed.tolist()}
    save("primary-cluster-subgroups.json", summary)
    mask = np.asarray(world_tour)
    selected, weights = sums[:, mask], counts[mask]
    random = np.random.default_rng(20261116)
    draws = []
    for _ in range(5000):
        clusters = random.integers(0, len(weights), len(weights))
        seeds = random.integers(0, len(selected), len(selected))
        draws.append(selected[seeds][:, clusters].sum(axis=1).mean(axis=0) / weights[clusters].sum())
    intervals = np.quantile(draws, [.025, .975], axis=0)
    result = {"kind": "Exploratory subgroup of the already-used primary replay; not fresh evidence or multiplicity-adjusted",
              "source": str(source.relative_to(REPO_ROOT)), "n": int(weights.sum()), "clusters": len(weights),
              "seeds": data["seeds"], "bootstrap_seed": 20261116, "replicates": len(draws),
              "deltas": {name: {"mean": summary["world_tour"]["mean_deltas"][i],
                                 "crossed_seed_tournament_ci95": intervals[:, i].tolist()}
                         for i, name in enumerate(data["metrics"])}}
    save("primary-world-tour-intervals.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
