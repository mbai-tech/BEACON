# Environment Generator

This folder generates planning environments as both JSON scene files and PNG
images.

## Setup

From the repo root:

```bash
cd /Users/ishita/Documents/GitHub/SURP/enviornment
python3 -m venv .venv-mac
source .venv-mac/bin/activate
pip install -r requirements.txt
```

## Generate All Families

To generate 100 scenes for each family:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run.py
```

This writes files to:

- `enviornment/data/images/`
- `enviornment/data/scenes/`

Important:

- Running `run.py` clears the existing files in those two `data/` folders first.
- Output filenames include the seed, so each file can be reproduced later.

Example output names:

```text
enviornment/data/images/sparse_clutter_000_seed123456789.png
enviornment/data/scenes/sparse_clutter_000_seed123456789.json
```

## Generate One Family

To generate one scene for a single family:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run_family.py sparse_clutter
```

Valid family names:

- `sparse_clutter`
- `dense_clutter`
- `narrow_passage`
- `semantic_trap`
- `perturbed`

This writes to:

- `enviornment/data/images/<family>_scene.png`
- `enviornment/data/scenes/<family>_scene.json`

Running `run_family.py` without a seed also clears the normal `data/` output
folders first.

## Save A Specific Seed

To regenerate the exact same environment from a known seed and save it in a
separate folder that is not cleared by `run.py`:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run_family.py sparse_clutter 12345
```

This writes to:

- `enviornment/saved_enviornments/images/`
- `enviornment/saved_enviornments/scenes/`

Example saved files:

```text
enviornment/saved_enviornments/images/sparse_clutter_seed12345.png
enviornment/saved_enviornments/scenes/sparse_clutter_seed12345.json
```

These saved environments are not deleted when you rerun `python3 enviornment/run.py`.

If you save the same seed again, a numbered suffix is added instead of
overwriting the previous saved copy.

## Reproduce From A Seed In Batch Output

If you see a batch-generated filename like:

```text
sparse_clutter_003_seed1989186592.json
```

you can reproduce that same scene with:

```bash
python3 enviornment/run_family.py sparse_clutter 1989186592
```
