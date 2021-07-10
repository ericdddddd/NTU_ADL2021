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
    parser.add_argument("--delexlevel", default=2, type=int, required=False, help="0: no delex; 1: delex values in \"slots\"; 2: delex values in both \"slots\" and \"actions\"")
    parser.add_argument("--target", default="/content/simpletod/", type=str, required=False, help="path to output")
    args = parser.parse_args()

    datafolder = args.data
    targetfolder = args.target
    for folder in ["train"]:
        if not os.path.exists(targetfolder + folder):
            os.makedirs(targetfolder + folder)
        inlmb = []
        fns = os.listdir(datafolder + folder)
        fns.sort()
        for fn in fns:
            print("\r" + fn,end='')
            if not fn.startswith("dialogue"):
                with open(datafolder + folder + "/" + fn, "r", encoding='utf8') as f:
                    data = json.load(f)
                with open(targetfolder + folder + "/" + fn, "w", encoding='utf8') as f:
                    json.dump(data, f, indent=1)
                continue
            with open(datafolder + folder + "/" + fn, "r", encoding='utf8') as f:
                data = json.load(f)
            i = 0
            while i < len(data):
                
                if not data[i]["dialogue_id"].endswith(".json"):
                    dbs = []
                    slots = {}
                    canmap = {}
                    vmap = {}
                    for j in range(len(data[i]["turns"])):
                        if data[i]["turns"][j]["speaker"] != "SYSTEM":
                            continue
                        if "service_results" in data[i]["turns"][j]["frames"][0]:
                            dbs += data[i]["turns"][j]["frames"][0]["service_results"]
                        if len(data[i]["turns"][j]["frames"][0]["slots"]) != 0:
                            slots = {}
                        for k in range(len(data[i]["turns"][j]["frames"][0]["actions"])):
                            assert(len(data[i]["turns"][j]["frames"][0]["actions"][k]["canonical_values"]) == len(data[i]["turns"][j]["frames"][0]["actions"][k]["values"]))
                            for l in range(len(data[i]["turns"][j]["frames"][0]["actions"][k]["canonical_values"])):
                                canmap[data[i]["turns"][j]["frames"][0]["actions"][k]["values"][l]] = data[i]["turns"][j]["frames"][0]["actions"][k]["canonical_values"][l]
                                vmap[data[i]["turns"][j]["frames"][0]["actions"][k]["canonical_values"][l]] = data[i]["turns"][j]["frames"][0]["actions"][k]["values"][l]
                        for k in range(len(data[i]["turns"][j]["frames"][0]["slots"])):
                            s = data[i]["turns"][j]["frames"][0]["slots"][k]["slot"]
                            slots[s] = data[i]["turns"][j]["utterance"][data[i]["turns"][j]["frames"][0]["slots"][k]["start"]:data[i]["turns"][j]["frames"][0]["slots"][k]["exclusive_end"]]
                        db = {}
                        for k in range(len(dbs)):
                            matched = True
                            for s in slots:
                                if s not in dbs[k]:
                                    matched = False
                                    break
                                if dbs[k][s] != canmap[slots[s]]:
                                    matched = False
                                    break
                            if matched:
                                db = copy.deepcopy(dbs[k])
                                for s in db:
                                    if db[s] in vmap:
                                        db[s] = vmap[db[s]]
                                break
                    
                    for j in range(1, len(data[i]["turns"]), 2):
                        domain = data[i]["turns"][j]["frames"][0]["service"].split("_")[0].lower()
                        assert(data[i]["turns"][j]["speaker"] == "SYSTEM")
                        assert(len(data[i]["turns"][j]["frames"]) == 1)
                        slots = copy.deepcopy(data[i]["turns"][j]["frames"][0]["slots"])
                        slots.sort(key = lambda x : -x["start"])
                        delex = data[i]["turns"][j]["utterance"]
                        delexed = set()
                        
                        if args.delexlevel >= 1:
                            for k in range(1, len(slots)):
                                assert(slots[k-1]["start"] >= slots[k]["exclusive_end"])
                            for k in range(len(slots)):
                                domain_slot = domain + "_" + slots[k]["slot"]
                                delex = delex[:slots[k]["start"]] + "[" + domain_slot + "]" + delex[slots[k]["exclusive_end"]:]
                                delexed.add(domain_slot)
                        if args.delexlevel >= 2:
                            slots2 = copy.deepcopy(data[i]["turns"][j]["frames"][0]["actions"])
                            slots2 = [x for x in slots2 if len(x["values"]) > 0]
                            slots2.sort(key = lambda x : -len(x["values"][0]))
                            for k in range(len(slots2)):
                                domain_slot = domain + "_" + slots2[k]["slot"]
                                if domain_slot in delexed:
                                    continue
                                for l in range(len(slots2[k]["values"])):
                                    delex = delex.replace(slots2[k]["values"][l], "[" + domain_slot + "]")
                                delexed.add(domain_slot)
                        
                        target = ''
                        belief = []
                        for k in range(len(data[i]["turns"][j-1]["frames"])):
                            for slot in data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"]:
                                belief += [[data[i]["turns"][j-1]["frames"][k]["service"].split("_")[0].lower(), slot, data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"][slot]]]
                        belief.sort(key = lambda x : x[0] + " " + x[1])
                        for k in range(len(belief)):
                            belief[k][2].sort()
                            belief[k][2] = belief[k][2][0]
                        belief = [x[0] + " " + x[1] + " " + x[2] for x in belief]
                        target += '<|belief|> ' + ", ".join(belief) + ' <|endofbelief|> '
                        
                        action = copy.deepcopy(data[i]["turns"][j]["frames"][0]["actions"])
                        action.sort(key = lambda x : x["act"])
                        action = [domain + " " + x["act"].lower() + " " + x["slot"] for x in action]
                        
                        targetaug = []
                        if "beginning" in data[i]["turns"][j].keys() and "end" in data[i]["turns"][j].keys():
                        
                            for k in range(len(data[i]["turns"][j]["beginning"])):
                                if "social" in data[i]["turns"][j]["beginning"][k]["justification"] or "useful" in data[i]["turns"][j]["beginning"][k]["justification"]:                   
                                    targetaug += [target + '<|action|> ' + "chitchat, " + ", ".join(action) + ' <|endofaction|> ' + '<|response|> ' + data[i]["turns"][j]["beginning"][k]["candidate"].strip() + ' ' + delex + ' <|endofresponse|>']
                            for k in range(len(data[i]["turns"][j]["end"])):
                                if "social" in data[i]["turns"][j]["end"][k]["justification"] or "useful" in data[i]["turns"][j]["end"][k]["justification"]:
                                    targetaug += [target + '<|action|> ' + ", ".join(action) + ", chitchat" + ' <|endofaction|> ' + '<|response|> ' + delex + ' ' + data[i]["turns"][j]["end"][k]["candidate"].strip() + ' <|endofresponse|>']
                        
                        target += '<|action|> ' + ", ".join(action) + ' <|endofaction|> '
                        target += '<|response|> ' + delex + ' <|endofresponse|>'
                        context = '<|context|> '
                        for k in range(j):
                            if k % 2 == 0:
                                context += '<|user|> '
                            else:
                                context += '<|system|> '
                            context += data[i]["turns"][k]["utterance"] + " "
                        context += '<|endofcontext|>'
                        if len(targetaug) != 0: # chit chat 情況
                            for k in range(len(targetaug)):
                                inlmb += [(context + targetaug[k]).replace("\n", " ").replace("\r", "")]    
                        else: # 沒有 chit chat 情況
                            inlmb += [(context + target).replace("\n", " ").replace("\r", "")]
                            
                else:
                    for j in range(1, len(data[i]["turns"]), 2):
                        assert(data[i]["turns"][j]["speaker"] == "SYSTEM")
                        delex = data[i]["turns"][j]["utterance"]
                        target = ''
                        belief = []
                        for k in range(len(data[i]["turns"][j-1]["frames"])):
                            for slot in data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"]:
                                belief += [[data[i]["turns"][j-1]["frames"][k]["service"].split("_")[0].lower(), slot, data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"][slot]]]
                        belief.sort(key = lambda x : x[0] + " " + x[1])
                        for k in range(len(belief)):
                            belief[k][2].sort()
                            belief[k][2] = belief[k][2][0]
                        belief = [x[0] + " " + x[1] + " " + x[2] for x in belief]
                        target += '<|belief|> ' + ", ".join(belief) + ' <|endofbelief|> '
                        targetaug = []
                        
                        if "beginning" in data[i]["turns"][j].keys() and "end" in data[i]["turns"][j].keys():
                        
                            for k in range(len(data[i]["turns"][j]["beginning"])):
                                if "social" in data[i]["turns"][j]["beginning"][k]["justification"] or "useful" in data[i]["turns"][j]["beginning"][k]["justification"]:
                                    targetaug += [target +  '<|response|> ' + data[i]["turns"][j]["beginning"][k]["candidate"].strip() + ' ' + delex + ' <|endofresponse|>']
                            for k in range(len(data[i]["turns"][j]["end"])):
                                if "social" in data[i]["turns"][j]["end"][k]["justification"] or "useful" in data[i]["turns"][j]["end"][k]["justification"]:
                                    targetaug += [target + '<|response|> ' + delex + ' ' + data[i]["turns"][j]["end"][k]["candidate"].strip() + ' <|endofresponse|>']
                                    
                        target += '<|response|> ' + delex + ' <|endofresponse|>'
                        context = '<|context|> '
                        for k in range(j):
                            if k % 2 == 0:
                                context += '<|user|> '
                            else:
                                context += '<|system|> '
                            context += data[i]["turns"][k]["utterance"] + " "
                        context += '<|endofcontext|>'
                        if len(targetaug) != 0: # chit chat 情況
                            for k in range(len(targetaug)):
                                inlmb += [(context + targetaug[k]).replace("\n", " ").replace("\r", "")]
                        else: # 沒有 chit chat 情況
                            inlmb += [(context + target).replace("\n", " ").replace("\r", "")]
                i += 1
            
        
        with open("lm.input."+folder+".both.txt", "w", encoding='utf8') as f: #SimpleTOD extension (all responses)
            f.write('\n'.join(inlmb))
            
if __name__ == '__main__':
    random.seed(42)
    main()