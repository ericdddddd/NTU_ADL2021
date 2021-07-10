# Copyright (c) Facebook, Inc. and its affiliates.

import json
import os
import copy
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", default=False, type=bool, required=False, help="use all dialogues rather than only augmented dialogues")
    parser.add_argument("--data", default="/content/drive/MyDrive/ADL/final/data-0614/", type=str, required=False, help="path to SGD")
    args = parser.parse_args()

    datafolder = args.data
    # "dev","test_seen","test_unseen"
    fold = ["test_seen"]
    for folder in fold:
        inlme = []
        fns = os.listdir(datafolder + folder)
        fns.sort()
        for fn in fns:
            print(fn)
            if not fn.startswith("dialogue"):
                with open(datafolder + folder + "/" + fn, "r", encoding='utf8') as f:
                    data = json.load(f)
                continue
            with open(datafolder + folder + "/" + fn, "r", encoding='utf8') as f:
                data = json.load(f)
            i = 0
            while i < len(data):
                
                for j in range(1, len(data[i]["turns"]), 2):
                    context = '<|context|> '
                    for k in range(j):
                        if k % 2 == 0:
                            context += '<|user|> '
                        else:
                            context += '<|system|> '
                        context += data[i]["turns"][k]["utterance"] + " "
                    context += '<|endofcontext|>'
                    if j >= 1:
                        inlme += [(context).replace("\n", " ").replace("\r", "")]
                    
                i += 1
 
        with open("lm.input."+folder+".txt", "w", encoding='utf8') as f: #used as the input during evaluation of SimpleTOD and SimpleTOD extension
            f.write('\n'.join(inlme))

if __name__ == '__main__':
    random.seed(42)
    main()