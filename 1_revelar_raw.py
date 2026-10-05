from src import config
from src.raw_convert import convert_arw_to_png

if __name__ == "__main__":
    for dataset in config.DATASETS:
        print(f"\n========== {dataset['name']} ==========")
        convert_arw_to_png(dataset["raw"], dataset["png"], config.RAW_MAX_DIM, config.RAW_ANONYMIZE,
                           config.RAW_BITS, mapping_csv=dataset["results"] / "0_mapeamento_nomes.csv")
