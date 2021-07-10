import json
from pathlib import Path
from torch.utils.data import Dataset 
from tqdm import tqdm
from transformers import BertTokenizer, BertForMultipleChoice
import torch
import random

def read_data(data_dir, mode='train'):
    data = []
    if mode == 'train':
        num_dial = 139
    elif mode == 'dev':
        num_dial = 21
    elif mode == 'test_seen':
        num_dial = 17
    elif mode == 'test_unseen':
        num_dial = 6
    
    # print("Data Path: ", mode)
    for i in range(1, num_dial):
        # if mode == 'test' or 'test_unseen':
        #     path = str(data_dir) + '/'
        # else:
        #     path = str(data_dir) + '/' + mode
        path = str(data_dir) + '/' + mode
        if i <10:
            path += '/dialogues_00'+str(i)+'.json'
        elif i>=10 and i<100:
            path += '/dialogues_0'+str(i)+'.json'
        else:
            path += '/dialogues_'+str(i)+'.json'
        with open(Path(path)) as f:
            data += json.load(f)

    return data

class ServDataset(Dataset):
    def __init__(self, data, serv_dict, mode):
        self.data = data
        self.serv_dict = serv_dict
        self.mode = mode
    def __getitem__(self,index):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dialog_id = self.data[index]["dialogue_id"]
        turn_id = self.data[index]["turn_id"]
        utterance = self.data[index]["utterance"]
        services = self.data[index]["services"]
        prompt = [ utterance for i in range(len(services))]
        # print(prompt)
        choice = [ self.serv_dict[serv] for serv in services ]
        # print(choice)
        encoding = tokenizer(prompt,choice, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
        return  dialog_id, turn_id, encoding, services
    def __len__(self):
        return len(self.data)

def extract_need_data(data):
    turns = [] # [{"dialogue_id":int, "service":str, "turn_id":int, "utterance":str}]
    no_frame_data = 0
    num_nomatch_turn = 0
    # no_frame_turn = 0
    for d in tqdm(data, desc="Extract data"):
        no_frame = False
        temp_turns = []
        for turn_id in range(0,len(d['turns']),2):
            temp_turn = {}
            if d['turns'][turn_id]['frames'] == []:
                no_frame = True
                break
            else:
                temp_turn['dialogue_id'] = d['dialogue_id']
                temp_turn['services'] = []
                temp_turn['services'].append(d['turns'][turn_id]['frames'][0]['service'])
                for serv in d['services']:
                    if serv != d['turns'][turn_id]['frames'][0]['service']:
                        temp_turn['services'].append(serv)
                temp_turn['utterance'] = d['turns'][turn_id]['utterance'] + " " + d['turns'][turn_id+1]['utterance']
                temp_turn['turn_id'] = [turn_id, turn_id+1]
                temp_turns.append(temp_turn)
        if no_frame == True:
            no_frame_data += 1
        elif no_frame == False:
            turns += temp_turns
    print("Number of no frame data: ", no_frame_data)

    return turns

# def extract_test_data(data):
#     turns = [] # [{"dialogue_id":int, "service":str, "turn_id":int, "utterance":str}]
#     # no_frame_turn = 0
#     for d in tqdm(data, desc="Extract data"):
#         no_frame = False
#         temp_turns = []
#         for turn_id in range(0,len(d['turns']),2):
#             temp_turn = {}
#             temp_turn['dialogue_id'] = d['dialogue_id']
#             temp_turn['services'] = d['services']
#             temp_turn['utterance'] = d['turns'][turn_id]['utterance'] + " " + d['turns'][turn_id+1]['utterance']
#             temp_turn['turn_id'] = [d['turns'][turn_id]["turn_id"], d['turns'][turn_id+1]["turn_id"]]
#             temp_turns.append(temp_turn)
#         turns += temp_turns
#     return turns

def extract_test_data(data):
    turns = [] # [{"dialogue_id":int, "service":str, "turn_id":int, "utterance":str}]
    # no_frame_turn = 0
    for d in tqdm(data, desc="Extract data"):
        no_frame = False
        temp_turns = []
        for turn_id in range(0,len(d['turns']),2):
            if len(d['services']) > 1:
                temp_turn = {}
                temp_turn['dialogue_id'] = d['dialogue_id']
                temp_turn['services'] = d['services']
                temp_turn['utterance'] = d['turns'][turn_id]['utterance'] + " " + d['turns'][turn_id+1]['utterance']
                temp_turn['turn_id'] = [d['turns'][turn_id]["turn_id"], d['turns'][turn_id+1]["turn_id"]]
                temp_turns.append(temp_turn)
        turns += temp_turns
    return turns

def postprocess(pred_dict, test_data_dir, save_result_dir, mode):
    if mode == "test_seen":
        num_dial = 17
    elif mode == "test_unseen":
        num_dial = 6
    for i in range(1, num_dial):
    # for i in range(6, num_dial):
        # read file
        data = []
        path = str(test_data_dir) + '/' + mode
        # path = str(test_data_dir)
        file_name = ''
        if i <10:
            file_name = '/dialogues_00'+str(i)+'.json'
            path += file_name
        elif i>=10 and i<100:
            file_name = '/dialogues_0'+str(i)+'.json'
            path += file_name
        else:
            file_name = '/dialogues_'+str(i)+'.json'
            path += file_name
        with open(Path(path)) as f:
            data = json.load(f)
        # processing
        for i in range(len(data)):
            dial_id = data[i]["dialogue_id"]
            for j in range(len(data[i]["turns"])):
                try:
                    if len(data[i]["services"]) == 1:
                        data[i]["turns"][j]['service'] = data[i]["services"][0]
                    else:
                        turn_id = data[i]["turns"][j]['turn_id']
                        data[i]["turns"][j]['service'] = pred_dict[dial_id][turn_id]
                except:
                    if len(data[i]["services"]) == 1:
                        print("Fail to assign service to dialog {} turn {}".format(dial_id))
                    else:
                        print("Fail to assign service to dialog {} turn {}".format(dial_id, turn_id))
                    continue
        # save file
        save_path = Path(str(save_result_dir) + file_name)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)
            # f.write(data)

    print("Processed test data was saved!")

def re_dataid_check(data_dir_path, save_dir_path, mode):
    # print("-------------------- {} data --------------------".format(mode))
    try:
        if mode == 'train':
            num_dial = 139
        elif mode == 'dev':
            num_dial = 21
        elif mode == 'test_seen' or mode == 'test_seen_servid':
            num_dial = 17
        elif mode == 'test_unseen' or mode == 'test_unseen_servid':
            num_dial = 6
    except:
        print("Cannot process {} dataset!".format(mode))
        return
    first = random.randint(1,100)
    num_no_frame_dialog = 0
    num_dialog_bef = 0
    num_dialog_aft = 0
    # print(num_dial)
    for i in tqdm(range(1, num_dial), desc="Reid data & check empty frame"):
        dialogs_processed = []
        secd = random.randint(1,50)
        # read data
        path = str(data_dir_path) + '/' + mode
        if i <10:
            file_name = '/dialogues_00'+str(i)+'.json'
            path += file_name
        elif i>=10 and i<100:
            file_name = '/dialogues_0'+str(i)+'.json'
            path += file_name
        else:
            file_name = '/dialogues_'+str(i)+'.json'
            path += file_name
        with open(Path(path)) as f:
            dialogs = json.load(f)
        # print(len(dialogs))
        # reid
        for j in range(len(dialogs)): 
            num_dialog_bef += 1
            old_dial_id = dialogs[j]["dialogue_id"]
            if secd <10:
                new_dial_id = str(first) + "_0000" + str(secd)
            elif secd>=10 and secd<100:
                new_dial_id = str(first) + "_000" + str(secd)
            elif secd>=100 and secd<1000:
                new_dial_id = str(first) + "_00" + str(secd)
            elif secd>=1000 and secd<10000:
                new_dial_id = str(first) + "_0" + str(secd)
            else:
                new_dial_id = str(first) + "_" + str(secd)
            secd += random.randint(1,5)
            dialogs[j]["old_dialogue_id"] = old_dial_id
            dialogs[j]["dialogue_id"] = new_dial_id

            # check frame
            if mode == "test_seen_servid" or mode == "test_unseen_servid":
                dialogs_processed.append(dialogs[j])
            else:
                no_frame = False
                for turn in dialogs[j]["turns"]:
                    if turn["frames"] == []:
                        no_frame = True
                        break
                if no_frame == False:
                    dialogs_processed.append(dialogs[j])
                else:
                    num_no_frame_dialog += 1

        first += random.randint(1,10)
        num_dialog_aft += len(dialogs_processed)

        # save file
        save_path = Path(str(save_dir_path) + file_name)
        with open(save_path, 'w') as f:
            json.dump(dialogs_processed, f, indent=4, sort_keys=True)
        # if dialogs_processed != []
        #     save_path = Path(str(save_dir_path) + file_name)
        #     with open(save_path, 'w') as f:
        #         json.dump(dialogs_processed, f, indent=4, sort_keys=True)
    if mode != "test_seen":
        print("Number of dialogues with empty frame: ", num_no_frame_dialog)
    elif mode != "test_unseen":
        print("Number of dialogues with empty frame: ", num_no_frame_dialog)
    print("Number of data before process: ", num_dialog_bef)
    print("Number of data after processed: ", num_dialog_aft)

def check_schema(data_dir_path, save_dir_path):
    schema_path = Path(data_dir_path/"schema.json")
    with open(schema_path,'r') as f:
        schema = json.load(f)
    for i, serv in enumerate(tqdm(schema, desc="Check schema")):
        for j, slot in enumerate(serv["slots"]):
            if "possible_values" not in list(slot.keys()):
                schema[i]["slots"][j]["is_categorical"] = False
            else:
                if slot["possible_values"] == []:
                    schema[i]["slots"][j]["is_categorical"] = False
                else:
                    schema[i]["slots"][j]["is_categorical"] = True    
    save_path = Path(save_dir_path/"schema.json")
    with open(save_path, 'w') as f:
        json.dump(schema, f, indent=4, sort_keys=True)   



# def re_dataid(data_dir_path, save_dir_path, mode):
#     if mode == 'train':
#         num_dial = 139
#     elif mode == 'dev':
#         num_dial = 21
#     elif mode == 'test':
#         num_dial = 17
#     first = random.randint(1,100)
#     for i in range(1, num_dial):
#         secd = random.randint(1,50)
#         # read data
#         path = str(data_dir_path) + '/' + mode
#         if i <10:
#             file_name = '/dialogues_00'+str(i)+'.json'
#             path += file_name
#         elif i>=10 and i<100:
#             file_name = '/dialogues_0'+str(i)+'.json'
#             path += file_name
#         else:
#             file_name = '/dialogues_'+str(i)+'.json'
#             path += file_name
#         with open(Path(path)) as f:
#             dialogs = json.load(f)
#         # print(len(dialogs))
#         # reid
#         for j in range(len(dialogs)): 
#             old_dial_id = dialogs[j]["dialogue_id"]
#             if secd <10:
#                 new_dial_id = str(first) + "_0000" + str(secd)
#             elif secd>=10 and secd<100:
#                 new_dial_id = str(first) + "_000" + str(secd)
#             elif secd>=100 and secd<1000:
#                 new_dial_id = str(first) + "_00" + str(secd)
#             elif secd>=1000 and secd<10000:
#                 new_dial_id = str(first) + "_0" + str(secd)
#             else:
#                 new_dial_id = str(first) + "_" + str(secd)
#             secd += random.randint(1,5)
#             dialogs[j]["old_dialogue_id"] = old_dial_id
#             dialogs[j]["dialogue_id"] = new_dial_id
#         first += random.randint(1,10)

#         # save file
#         # print(len(dialogs),"\n")
#         save_path = Path(str(save_dir_path) + file_name)
#         with open(save_path, 'w') as f:
#             json.dump(dialogs, f, indent=4, sort_keys=True)
#             # f.write(data)

# def check_schema(data_dir_path, save_dir_path):
#     schema_path = Path(data_dir_path/"schema.json")
#     with open(schema_path,'r') as f:
#         schema = json.load(schema_path)
#     for i, serv in enumerate(schema):
#         for j, slot in enumerate(serv["slots"]):
#             if slot["possible_value"] == []:
#                 schema[i]["slots"][j]["is_categorical"] = False
#             else:
#                 schema[i]["slots"][j]["is_categorical"] = True    
#     save_path = Path(save_dir_path/"schema.json")
#         with open(save_path, 'w') as f:
#             json.dump(schema, f, indent=4, sort_keys=True)   


# def re_dataid(dialogs):
#     first = 1
#     secd = 0
#     for i in range(len(dialogs)): 
#         old_dial_id = dialogs[i]["dialogue_id"]
#         if secd <10:
#             new_dial_id = str(first) + "_0000" + str(secd)
#         elif secd>=10 and secd<100:
#             new_dial_id = str(first) + "_000" + str(secd)
#         elif secd>=100 and secd<1000:
#             new_dial_id = str(first) + "_00" + str(secd)
#         elif secd>=1000 and secd<10000:
#             new_dial_id = str(first) + "_0" + str(secd)
#         else:
#             new_dial_id = str(first) + "_" + str(secd)
#         # print(new_dial_id)
#         secd += 1
#         if secd > 1000:
#             secd = 0
#             first += 1
#             # print(new_dial_id)
#         dialogs[i]["old_dialogue_id"] = old_dial_id
#         dialogs[i]["dialogue_id"] = new_dial_id

#     return dialogs

class Clflabel:
    def __init__(self, label_file):
        self.label_file = label_file

    def collect_serv(self):
        serv_dict = {}
        for serv in self.label_file:
            serv_dict[serv['service_name']] = serv['description']
        return serv_dict
    
    def collect_label(self):
        self.label_dict = {} # dict[service_na] = {dict[slot_na]:possible value}
        for label in self.label_file:
            serv = {}
            serv_na = label['service_name']
            temp_slot = {}
            for slot in label['slots']:
                try:
                    temp_slot[slot['name']] = slot['possible_values']
                except:
                    temp_slot[slot['name']] = []
            self.label_dict[serv_na] = temp_slot

        return self.label_dict

    def extract_need_schema(self):
        schema = []
        for serv in self.label_file:
            temp_serv = {}
            temp_serv["service_name"] = serv["service_name"]
            temp_serv["description"] = serv["description"]
            temp_serv["slots"] = serv["slots"]
            temp_serv["intents"] = []
            for intent in serv["intents"]:
                temp_intent = {}
                temp_intent["name"] = intent["name"]
                temp_intent["description"] = intent["description"]
                temp_serv["intents"].append(temp_intent)
            schema.append(temp_serv)
        return schema
    
    def index_dict(self):
        self.slotToid_dict = {} # dict[serv_na][slot_na] = slot_id
        self.servToid_dict = {} # dict[serv_na] = serv_id
        self.idToslot_dict = {} # dict[serv_id][slot_id] = slot_na
        self.idToserv_dict = {} # dict[serv_id] = serv_na
        for i,serv_key in enumerate(sorted(list(self.label_dict.keys()))): 
            self.servToid_dict[serv_key] = i
            self.idToserv_dict[i] = serv_key
            self.slotToid_dict[serv_key] = {}
            self.idToslot_dict[serv_key] = {}
            for j,slot_key in enumerate(sorted(list(self.label_dict[serv_key].keys()))):
                self.slotToid_dict[serv_key][slot_key] = j
                self.idToslot_dict[i][j] = slot_key

    def find_max_num_slot(self):
        max_num_slot = 0
        for serv_key in list(self.label_dict.keys()):
            num_slot = len(self.label_dict[serv_key])
            if num_slot > max_num_slot:
                max_num_slot = num_slot
        return max_num_slot
    
    def id_to_serv(self, serv_id):
        return self.idToserv_dict[serv_id]
    def serv_to_id(self, serv_na):
        return self.servToid_dict[serv_na]
    def slot_to_id(self, serv_na, slot_na):
        return self.slotToid_dict[serv_na][slot_na]
    def id_to_slot(self, serv_id, slot_id):
        return self.idToslot_dict[serv_id][slot_id]


def schema_embedding(schema, label_dict):
    intent_dict = dict()
    slot_dict = dict()
    non_slot_dict = dict()
    slot_pair = dict()
    cnt = 0
    for item in schema:
        tmp_slot = list()
        tmp_slot_desc = list()
        check_cat = list()
        service_desc = item['description']
        tmp_list = list()
        for intent in item['intents']:
            intent_dict[intent['name']] = '[CLS]' + service_desc + '[SEP]' + intent['description'] + '[SEP]'
        for slot in item['slots']:
            if slot['is_categorical']:
                slot_dict[slot['name']] = '[CLS]' + service_desc + '[SEP]' + slot['description'] + '[SEP]'
                for value in slot['possible_values']:
                    slot_pair[slot['name']+'_'+value] = '[CLS]' + slot['description'] + '[SEP]' + value
            else:
                non_slot_dict[slot['name']] = '[CLS]' + service_desc + '[SEP]' + slot['description'] + '[SEP]'
    
    
    
            



        

    




        
