# TEASE

TEASE is a lightweight Python library for closed-form item-item recommendation with approximate Thompson-style exploration.

It builds on EASE and adds Gaussian posterior score sampling for recommendation-time exploration while keeping the implementation compact and fast.

## Features

- Closed-form EASE recommender for implicit-feedback item-item ranking.
- TEASE exploration variant with Gaussian score sampling.
- Low-rank EASE and low-rank TEASE variants for lower serving memory.
- Simple sparse-matrix API built on NumPy and SciPy.
- Model save and load support.

## Installation

This repository uses a modern `pyproject.toml`-based build.

Install from a local checkout:

```bash
pip install .
```

Install in editable mode for development:

```bash
pip install -e .
```

Install directly from GitHub:

```bash
pip install "git+https://github.com/drdakjak/tEASE.git"
```

Build a distributable package:

```bash
python -m build
```

Project metadata is defined in [pyproject.toml](pyproject.toml).

## Quick Start

```python
import numpy as np
from scipy.sparse import csr_matrix

from tease import TEASE

X = csr_matrix(
    [
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ],
    dtype=np.float64,
)

model = TEASE(l2_scale=0.3, random_state=7).fit(X)
user_vector = csr_matrix([[1, 0, 0, 0, 0]], dtype=np.float64)
recommendations = model.recommend(
    user_vector,
    n=3,
    exploration_scale=0.25,
    mask_seen=True,
)

print(recommendations)
```

## API

### `EASE`

Deterministic closed-form item-item recommender.

Important hyperparameters:

- `l2_scale`: scale-adaptive regularization strength.
  Reasonable values: `0.03`, `0.1`, `0.3`, `1.0`, `3.0`.
  Use smaller values for sharper recommendations and larger values for more conservative rankings.
- `compute_dtype`: fitting dtype.
  Reasonable values: `np.float64`, `np.float32`.
  Prefer `np.float64` during fit.
- `storage_dtype`: fitted parameter dtype.
  Reasonable values: `None`, `np.float32`, `np.float64`.
  Use `np.float32` to reduce model size after fitting.

### `TEASE`

Approximate Thompson-style exploration on top of EASE via Gaussian score sampling.

Important hyperparameters:

- `l2_scale`: inherited regularization strength.
  Reasonable values: `0.03`, `0.1`, `0.3`, `1.0`, `3.0`.
- `random_state`: reproducibility control.
  Set it during experiments and evaluation when you want stable exploratory rankings.
- `exploration_scale`: ranking-time exploration intensity.
  Reasonable values: `0.0`, `0.1`, `0.25`, `0.5`, `1.0`, `1.5`.
  Use `0.0` for deterministic EASE behavior, `0.1`-`0.5` as a practical production range, and values above `1.0` only when you intentionally want aggressive exploration.

### `LowRankEASE`

Approximate EASE with lower serving-time memory.

Important hyperparameters:

- `rank`: number of retained eigenvectors.
  Reasonable values: `50`, `100`, `200`, `300`, `500`.
  Start around `200`, increase rank for better approximation quality, and decrease it to reduce memory.
- `l2_scale`: inherited regularization strength.
  Reasonable values: `0.03`, `0.1`, `0.3`, `1.0`, `3.0`.

### `LowRankTEASE`

Low-rank TEASE with approximate exploration and diagonal-corrected uncertainty estimates.

Important hyperparameters:

- `rank`: number of retained eigenvectors.
  Reasonable values: `50`, `100`, `200`, `300`, `500`.
- `l2_scale`: inherited regularization strength.
  Reasonable values: `0.03`, `0.1`, `0.3`, `1.0`, `3.0`.
- `exploration_scale`: ranking-time exploration intensity.
  Reasonable values: `0.0`, `0.1`, `0.25`, `0.5`, `1.0`.

## Design Notes

TEASE is best described as approximate Thompson-style exploration rather than exact Thompson sampling. The exploration step perturbs EASE scores with a Gaussian approximation derived from the inverse regularized Gram matrix.

This makes the library a good fit when you want:

- strong EASE-style recommendation quality,
- simple closed-form training,
- uncertainty-aware score sampling at serving time,
- and low operational complexity.

## Development

Run a quick local import check after installation:

```bash
python -c "from tease import EASE, TEASE, LowRankEASE, LowRankTEASE; print('ok')"
```

Install development tooling when you need to build or publish releases:

```bash
pip install -e .[dev]
```
