# MVFS Lab

This repository contains multiple multi-view unsupervised feature selection methods in a single monorepo.

Current projects:

- `uimufslr`
- `teruimufs`

Each project keeps its own original data contract and internal implementation.  
The `mvfs_core` layer only provides a common way to register, run, and test projects.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements-dev.txt