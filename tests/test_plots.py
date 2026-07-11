import importlib.util

import pytest

from instella_reasoning.analysis import plots


def test_categorical_palette_is_fixed_length() -> None:
    assert len(plots.CATEGORICAL) == 8
    assert len(set(plots.CATEGORICAL)) == 8  # all distinct


def test_quadrant_of_labels() -> None:
    assert plots.quadrant_of(0.9, 0.9) == "genuine"
    assert plots.quadrant_of(0.9, 0.1) == "fragile"
    assert plots.quadrant_of(0.1, 0.9) == "consistent_gap"
    assert plots.quadrant_of(0.1, 0.1) == "random"


def test_status_and_contamination_colors_present() -> None:
    assert set(plots.STATUS) == {"genuine", "partial", "fragile", "gap"}
    assert set(plots.CONTAMINATION_COLOR) == {"contaminated", "partial", "clean"}


def test_require_matplotlib_message_without_extra() -> None:
    if importlib.util.find_spec("matplotlib") is not None:
        pytest.skip("matplotlib installed; graceful-failure path not exercised")
    with pytest.raises(RuntimeError, match="viz extra"):
        plots._require_matplotlib()


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib not installed"
)
def test_reliability_atlas_renders_when_available(tmp_path) -> None:
    from instella_reasoning.analysis.atlas import AtlasCell, AtlasReport

    report = AtlasReport(
        cells=[
            AtlasCell("arithmetic", "all", 5, 0.8, 0.4, 0.32, "fragile"),
            AtlasCell("logic", "all", 5, 0.6, 0.7, 0.42, "partial"),
        ]
    )
    out = plots.plot_reliability_atlas(report, tmp_path / "atlas.png")
    assert out.exists()
