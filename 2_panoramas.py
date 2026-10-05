from src import config
from src.pipeline import run_pipeline

if __name__ == "__main__":
    for dataset in config.DATASETS:
        print(f"\n========== {dataset['name']} ==========")
        run_pipeline(dataset["png"], dataset["results"], dataset["projection"], dataset["beta"],
                     dataset["ba"], dataset["align"])
