import argparse

import json
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lang1', type=str, required=True, help='2 letter code of the language')
    parser.add_argument('--lang2', type=str, required=True, help='2 letter code of the language')
    parser.add_argument('--lang1_path', type=str, required=True, help='Path to langauge corpus')
    parser.add_argument('--lang2_path', type=str, required=True, help='Path to langauge corpus')
    parser.add_argument('--save_path', type=str, required=True, help='Where to save the json file')
    
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.lang1_path, 'r') as f:
        lang1 = [l.strip() for l in f.readlines()]
    
    with open(args.lang2_path, 'r') as f:
        lang2 = [l.strip() for l in f.readlines()]

    bitext = {'data': []}
    for i, (l1, l2) in enumerate(tqdm(zip(lang1, lang2))):
        bitext['data'].append({'id': i, 'translation': {f'{args.lang1}': l1, f'{args.lang2}': l2}})

    with open(args.save_path, 'w') as f:
        json.dump(bitext, f)

if __name__ == "__main__": 
     main() 
