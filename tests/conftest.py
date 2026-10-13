from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.fixture
def research_cfg(tmp_path):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        return compose(
            config_name="research",
            overrides=[
                f"output_dir={tmp_path}",
                "fit.max_epochs=1",
                "split.n_folds=2",
                "search.n_trials=1",
            ],
        )
