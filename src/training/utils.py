import random
import json
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results: dict, path: str):
    """Save results dict as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_multiseed(train_fn, config: dict, seeds: list, save_dir: str):
    """
    Run training for multiple seeds and collect results.

    Args:
        train_fn:  function(config, seed) -> dict with keys: 'history', 'fid', 'is_mean', 'is_std'
        config: training config dict
        seeds: list of seeds, e.g. [42, 123, 456]
        save_dir: directory to save per-seed results

    Returns:
        dict: {seed: result_dict}
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed}")
        print(f"{'='*50}")

        set_seed(seed)
        result = train_fn(config, seed)
        all_results[seed] = result

        # save individual seed result
        save_results(result, save_dir / f"seed_{seed}.json")
        print(f"Seed {seed} done — FID: {result.get('fid', 'N/A'):.2f}")

    # save all results together
    save_results(all_results, save_dir / "all_seeds.json")
    print(f"\nAll seeds done. Results saved to {save_dir}")
    return all_results
