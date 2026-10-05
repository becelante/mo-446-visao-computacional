"""Passo 3: extras X1 a X3 (bundle adjustment e projecao cilindrica ou esferica)
para cada conjunto de config.EXTRAS, em extras/<conjunto>/."""
from src import config
from src.passo3_pipeline_extras import run_pipeline

if __name__ == "__main__":
    for dataset in config.EXTRAS:
        if not dataset["png"].is_dir():
            print(f"Pasta {dataset['png'].name}/ nao encontrada (rode o passo 1), pulando.")
            continue
        print(f"\n========== {dataset['name']} ==========")
        run_pipeline(dataset["png"], dataset["results"], dataset["projection"], dataset["beta"],
                     dataset["ba"], dataset["align"])
