# VS Code Setup

Open this repository in VS Code, then create a local Mac virtual environment for the
`enviornment` folder:

```bash
cd /Users/ishita/Documents/GitHub/SURP/enviornment
python3 -m venv .venv-mac
source .venv-mac/bin/activate
pip install -r requirements.txt
```

After that, in VS Code:

1. Open the Run and Debug panel.
2. Choose `Run Circle Environment` or `Run Rectangle Environment`.
3. Click Run.

The generated images and scene JSON files will be written to `enviornment/data/`.
