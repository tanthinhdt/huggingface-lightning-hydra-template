"""python -m src.research_cli action=run"""

import hydra
from omegaconf import DictConfig

from src.research.workflows import run


@hydra.main(version_base="1.3", config_path="../configs", config_name="research")
def main(cfg: DictConfig):
    print(run(cfg))


if __name__ == "__main__":
    main()
