from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import pickle
from utils_unseen import read_data
from state_to_csv import write_csv


def main(args) -> None:
    result_path = args.data_dir/("pred_res_"+args.mode+"_domain/")
    data = read_data(result_path, mode = args.mode)
    ans = {}
    for d in data:
        dialogue_id = d['old_dialogue_id']
        tmp = {}
        for i, turn in enumerate(d['turns']):
            if i % 2 == 1:
                continue
            service = turn['frames'][0]['service']
            for slot, value in turn['frames'][0]['state']['slot_values'].items():
                tmp[service + '-' + slot] = value[0]
        ans[dialogue_id] = tmp
    
    write_csv(ans, args.result_file)
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./predictions/",
    )

    parser.add_argument(
        "--result_file",
        type=Path,
        help="Directory to the dataset.",
        default="./predictions/submission.csv",
    )

    parser.add_argument("--mode", type=str, default="test_seen")

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    main(args)
