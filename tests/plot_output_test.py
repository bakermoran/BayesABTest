"""Render real plots to disk so they can be inspected by eye."""

from pathlib import Path

import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent.parent / 'plots'


def test_save_posteriors_png(fitted_poisson_model):
    """Save plot_posteriors output to plots/ and check it was written.

    plots/ is git-ignored. Run `uv run pytest tests/plot_output_test.py`.
    """
    PLOT_DIR.mkdir(exist_ok=True)
    fitted_poisson_model.plot_posteriors()
    out = PLOT_DIR / 'posteriors.png'
    plt.savefig(out, dpi=100)
    plt.close('all')
    print(f'\nSaved plot to {out}')
    assert out.stat().st_size > 0
