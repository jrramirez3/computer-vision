

import json
import argparse
import os


def load_json(data_path, jsfile):
    with open(os.path.join(data_path, jsfile), 'r') as f:
        js = json.load(f)

    return js



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j",
                        "--json",
                        default='grab.json',
                        help='Json filename')
    parser.add_argument("-c",
                        "--csv",
                        default='labels_train.csv',
                        help='Csv filename')
    parser.add_argument("-p",
                        "--data_path",
                        default='dataset/grabngo/drinks',
                        help='Csv filename')
    args = parser.parse_args()

    js = load_json(args.data_path, args.json)
    print(js)


    

