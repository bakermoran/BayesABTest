"""Render real plots to disk so they can be inspected by eye."""

from pathlib import Path

import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent.parent / 'plots'


def test_save_report_png(fitted_poisson_result):
    """Save the full report to plots/ and check it was written.

    plots/ is git-ignored. Run `uv run pytest tests/plot_output_test.py`.
    """
    PLOT_DIR.mkdir(exist_ok=True)
    fig = fitted_poisson_result.plot(compare_variants=True)
    out = PLOT_DIR / 'report.png'
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f'\nSaved plot to {out}')
    assert out.stat().st_size > 0
