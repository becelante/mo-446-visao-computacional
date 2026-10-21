"""Passo 2: pipeline principal (Secoes 1 a 6 do relatorio) no conjunto D1, em D1_results/."""
from src import config
from src.passo2_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(config.BASE_INPUT, config.BASE_RESULTS)
