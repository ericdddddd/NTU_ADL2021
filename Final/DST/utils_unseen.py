import json
from pathlib import Path
from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

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
    for i in range(1, num_dial):
        if mode == 'test_seen' or mode == 'test_unseen':
            path = str(data_dir) + '/'
        else:
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


def extract_need_data(data):
    turns = [] # [{"dialogue_id":int, "service":str, "turn_id":int, "utterance":str}]
    no_frame_data = 0
    num_nomatch_turn = 0
    # no_frame_turn = 0
    for d in data:
        no_frame = False
        temp_turns = []
        for turn_id in range(0,len(d['turns']),2):
            temp_turn = {}
            if d['turns'][turn_id]['frames'] == []:
                no_frame = True
                break
            else:
                temp_turn['dialogue_id'] = d['dialogue_id']
                temp_turn['services'] = dict()
                for serv in d['services']:
                    temp_turn['services'][serv] = 0

                # all_null = True
                # for frame in d['turns'][turn_id]['frames']:
                #     if frame['actions'] != []:

                temp_turn['services'][d['turns'][turn_id]['frames'][0]['service']] = 1
                if d['turns'][turn_id+1]['frames'] != []:
                        temp_turn['services'][d['turns'][turn_id+1]['frames'][0]['service']] = 1
                # if all_null and not d['services']:
                #     temp_turn['services'][d['turns'][turn_id]['frames'][0]['service']] = 1

                user_utterance = d['turns'][turn_id]['utterance']
                system_utterance = d['turns'][turn_id+1]['utterance']
                temp_turn['utterance'] = user_utterance + ' ' + system_utterance
                temp_turn['turn_id'] = [turn_id, turn_id + 1]
                
                temp_turns.append(temp_turn)

                
        if no_frame == True:
            no_frame_data += 1
        elif no_frame == False:
            turns += temp_turns
        
    print("Number of no frame data: ", no_frame_data)

    return turns



def extract_test_data(data):
    turns = [] # [{"dialogue_id":int, "service":str, "turn_id":int, "utterance":str}]
    # no_frame_turn = 0
    for d in tqdm(data, desc="Extract data"):
        no_frame = False
        temp_turns = []
        for turn_id in range(0,len(d['turns']),2):
            temp_turn = {}
            temp_turn['dialogue_id'] = d['dialogue_id']
            temp_turn['services'] = d['services']
            user_utterance = d['turns'][turn_id]['utterance']           
            system_utterance = d['turns'][turn_id+1]['utterance']
            temp_turn['utterance'] = user_utterance + ' ' + system_utterance
            temp_turn['turn_id'] = [turn_id, turn_id + 1]
            temp_turns.append(temp_turn)
        turns += temp_turns
    return turns


def create_utterance(datas, schema, tokenizer, mode = 'train'):
    data_pair = []
    for id, data in enumerate(tqdm(datas, desc = "Extract utterance")):
        instance_set = list()
        dialogue_id = data['dialogue_id']
        turn_id = data['turn_id']
        utterance = data['utterance']

        if mode == 'train':
            service_set = list(data['services'].keys())
            labels = list(data['services'].values())
        else:
            service_set = data['services']
        for service in service_set:
            desc = schema[service]
            service_text = service + ' ' + desc            
            token = tokenizer(service_text, utterance, max_length = 128, padding = 'max_length', truncation = True, return_tensors="pt", return_token_type_ids = True)


            instance_set.append(token)
        
        if mode == 'train':
            pair = {
                'data_id' : dialogue_id,
                'turn_id' : turn_id,
                'text_set' : instance_set,
                'services' : service_set,
                'labels' : labels,
            }
        else:
            pair = {
                'data_id' : dialogue_id,
                'turn_id' : turn_id,
                'text_set' : instance_set,
                'services' : service_set,
            }
        data_pair.append(pair)

    return data_pair

def create_data(service_embedding, text_embedding, service_to_id, datas):
    data_pair = []
    for id, data in enumerate(datas):
        service_set = list()
        dialogue_id = data['dialogue_id']
        turn_id = data['turn_id']
        service_set = [service_embedding[service_to_id[service]].squeeze() for service in data['services']]
        text = text_embedding[dialogue_id].squeeze()
        pair = {
                'text' : text,
                'service' : service_set,
                'data_id' : dialogue_id,
                'turn_id' : turn_id,
        }
        data_pair.append(pair)
    return data_pair

def postprocess(pred_dict, test_data_dir, save_result_dir, mode):
    if mode == 'test_unseen':
        num_dial = 6
    else:    
        num_dial = 17
    for i in range(1, num_dial):
        # read file
        data = []
        path = str(test_data_dir)
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
                turn_id = data[i]["turns"][j]['turn_id']
                try:
                    data[i]["turns"][j]['service'] = pred_dict[dial_id][turn_id]
                except:
                    print("Fail to assign service to dialog {} turn {}".format(dial_id, turn_id))
                    continue
        # save file
        save_path = Path(str(save_result_dir) + file_name)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)
            # f.write(data)

    print("Processed test data was saved!")



class EmbedDataset(Dataset):
    def __init__(self,data):
        self.data = data
        
    def __getitem__(self,idx):

        text = self.data[idx]['text']
        text_ids = text['input_ids'].squeeze()
        text_mask = text["attention_mask"].squeeze()
        id = self.data[idx]['data_id']
        
        return text_ids, text_mask, id
    
    def __len__(self):
        return len(self.data)

class finalDataset(Dataset):
    def __init__(self,data, mode = 'train'):
        self.data = data
        self.mode = mode
        
    def __getitem__(self,idx):

        if self.mode == 'train':
            dialogue_id = self.data[idx]['data_id']
            turn_id = self.data[idx]['turn_id']
            text_set = self.data[idx]['text_set']
            # input_ids = text['input_ids'].squeeze()
            # token_type_ids = text['token_type_ids'].squeeze()
            # attention_mask = text['attention_mask'].squeeze()
            services = self.data[idx]['services']
            labels = self.data[idx]['labels']  

            return dialogue_id, turn_id, text_set, services, labels
        else:
            dialogue_id = self.data[idx]['data_id']
            turn_id = self.data[idx]['turn_id']
            text_set = self.data[idx]['text_set']
            # input_ids = text['input_ids'].squeeze()
            # token_type_ids = text['token_type_ids'].squeeze()
            # attention_mask = text['attention_mask'].squeeze()
            services = self.data[idx]['services']

            return dialogue_id, turn_id, text_set, services    
        
    def __len__(self):
        return len(self.data)
  


class DSTlabel:
    def __init__(self, label_file):
        self.label_file = label_file

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
            temp_serv["intents"] = serv["intents"]
            # temp_serv["intents"] = []
            # for intent in serv["intents"]:
            #     temp_intent = {}
            #     temp_intent["name"] = intent["name"]
            #     temp_intent["description"] = intent["description"]
            #     temp_serv["intents"].append(temp_intent)
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

    def get_service(self):
        service_dict = {}
        for item in self.label_file:
            service = item['service_name']
            service_dict[service] = item['description']
        return service_dict

def serv_pred(model, testloader, test_data, device):
    model.eval()
    pred = []
    dialog_id_list = list()
    turn_id_list = list()
    services_list = list()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(testloader)):
            dialogue_id, turn_id, text_set, services = batch
            services = test_data[batch_id]['services']     
            pred_list = list()
            for i, text in enumerate(text_set):
                input_ids = text['input_ids'].squeeze(0).to(device)
                token_type_ids = text['token_type_ids'].squeeze(0).to(device)
                attention_mask = text['attention_mask'].squeeze(0).to(device)
                output = model(input_ids, token_type_ids, attention_mask)
                output = output.squeeze(0)

                pred_list.append(int(torch.round(torch.sigmoid(output)).item()))

            pred_service = [services[i] for i,p in enumerate(pred_list) if p]
            if pred_service == []:
                pred_service = [services[0]]

            pred.append(pred_service)
            dialog_id_list += dialogue_id
            turn_id_list.append(turn_id)
            services_list.append(services)


    pred_dict = {} # {dialog_id:{turn_id:service}}
    for i in range(len(pred)):
        if dialog_id_list[i] not in pred_dict.keys():
            pred_dict[dialog_id_list[i]] = {}

        pred_dict[dialog_id_list[i]][turn_id_list[i][0].item()] = pred[i]
        pred_dict[dialog_id_list[i]][turn_id_list[i][1].item()] = pred[i]
    return pred_dict


    



    




        
