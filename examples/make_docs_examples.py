"""
Convert selected examples/*.py scripts to executed docs/*.ipynb notebooks
and regenerate docs/examples.rst.

Usage (run from any directory):
    python examples/make_docs_examples.py

Requirements:
    pip install nbformat nbconvert ipykernel
"""

import os
import json
import atexit
import shutil
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

# ---------------------------------------------------------------------------
# Python interpreter used to execute the notebooks.
# Change this to match your environment if needed.
# ---------------------------------------------------------------------------
PYTHON = os.path.expanduser("~/venv/oasys2/bin/python")

# ---------------------------------------------------------------------------
# Edit this list to control which examples appear in the docs.
# Each entry is (python_filename, title_in_docs).
# Paths are relative to the examples/ directory.
# ---------------------------------------------------------------------------
EXAMPLES = [
    # gol_examples.py excluded: plot_surface uses deprecated fig.gca(projection='3d')
    ("srfunc_examples.py",
     "Synchrotron source calculations with srfunc"),
    ("waveoptics/example_ideal_lens.py",
     "Waveoptics: focusing with an ideal lens"),
    ("waveoptics/example_arago_poisson.py",
     "Waveoptics: Arago-Poisson spot"),
]
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))        # examples/
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                       # project root
DOCS_DIR    = os.path.join(PROJECT_DIR, "docs")

# Prepended to every notebook before the example code.
SETUP_CELL = """\
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

try:
    import srxraylib.plot.gol as _gol
    _gol.set_qt = lambda: None
except Exception:
    pass
"""

# Temporary directory holding the custom kernel spec; cleaned up on exit.
_KERNEL_TMPDIR = tempfile.mkdtemp()
atexit.register(shutil.rmtree, _KERNEL_TMPDIR, ignore_errors=True)


def make_kernel_manager(python_path):
    """Return a KernelManager that launches the given Python executable directly."""
    kernel_dir = os.path.join(_KERNEL_TMPDIR, "custom_kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    with open(os.path.join(kernel_dir, "kernel.json"), "w") as f:
        json.dump({
            "argv": [python_path, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (custom)",
            "language": "python",
        }, f)

    ksm = KernelSpecManager()
    ksm.kernel_dirs = [_KERNEL_TMPDIR]
    return KernelManager(kernel_name="custom_kernel", kernel_spec_manager=ksm)


def py_to_executed_notebook(py_path, ipynb_path, title, timeout=300):
    """Read a Python script, wrap it in a notebook, execute it, save to disk."""
    with open(py_path) as f:
        source = f.read()

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(f"# {title}"),   # required by nbsphinx for toctree title
        nbformat.v4.new_code_cell(SETUP_CELL),
        nbformat.v4.new_code_cell(source),
    ]

    km = make_kernel_manager(PYTHON)
    client = NotebookClient(nb, timeout=timeout, km=km)
    # run with cwd = directory of the script so relative paths still work
    client.execute(cwd=os.path.dirname(py_path))

    with open(ipynb_path, "w") as f:
        nbformat.write(nb, f)


def write_examples_rst(examples, docs_dir):
    """Regenerate docs/examples.rst from the EXAMPLES list."""
    lines = [
        "========\n",
        "Examples\n",
        "========\n",
        "\n",
        ".. toctree::\n",
        "   :maxdepth: 1\n",
        "\n",
    ]
    for py_name, title in examples:
        nb_name = os.path.splitext(os.path.basename(py_name))[0] + ".ipynb"
        lines.append(f"   {title} <{nb_name}>\n")

    rst_path = os.path.join(docs_dir, "examples.rst")
    with open(rst_path, "w") as f:
        f.writelines(lines)
    print(f"  written {rst_path}")


if __name__ == "__main__":
    print("Converting examples to notebooks:")
    for py_name, title in EXAMPLES:
        py_path    = os.path.join(SCRIPT_DIR, py_name)
        ipynb_name = os.path.splitext(os.path.basename(py_name))[0] + ".ipynb"
        ipynb_path = os.path.join(DOCS_DIR, ipynb_name)
        print(f"  {py_name} -> docs/{ipynb_name} ... ", end="", flush=True)
        py_to_executed_notebook(py_path, ipynb_path, title)
        print("done")

    print("Updating docs/examples.rst:")
    write_examples_rst(EXAMPLES, DOCS_DIR)
    print("All done.")
