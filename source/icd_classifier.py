import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, GPT2Model, GPT2Tokenizer
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os, argparse
from prepare_data import prepare_data
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix

labels = {'material':0,
          'physics':1,
          'chemistry':2,
          'cs':3,
          'medical':4,
          'socialscience':5
          }


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, tokenizer, max_len=512):
        print("Reading lables")
        self.labels = torch.tensor(data_dict["label"])#[labels[label] label in data_dict['label']]
        print("Reading texts")
        self.texts = data_dict["text"]#[tokenizer(text, 
                     #          padding='max_length', max_length = max_len, truncation=True,
                     #           return_tensors="pt") for text in data_dict['text']]
        self.max_len = max_len
        print("Done Reading texts")

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return tokenizer(self.texts[idx], padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y



class BertClassifier(nn.Module):

    def __init__(self, HFmodel, emb_size=768, nclasses=6, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(HFmodel)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(emb_size, nclasses)
        self.sigmoid = nn.Sigmoid() #Sigmoid

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sigmoid(linear_output)

        return final_layer

class GptClassifier(nn.Module):
    def __init__(self, HFmodel, emb_size=768, seq_len=128, nclasses=6, dropout=0.5):
        super(GptClassifier, self).__init__()
        self.gpt = GPT2Model.from_pretrained(HFmodel)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(emb_size*seq_len, nclasses)

    def forward(self, input_id, mask):
        gpt_out, _ = self.gpt(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        dropout_output = self.dropout(gpt_out.view(batch_size,-1))
        linear_output = self.linear(dropout_output)
        return linear_output
        
def compute_accuracy(y, yhat):
    cm = multilabel_confusion_matrix(y,yhat)
    r = 1e-10
    dict_macro = {}
    dict_micro = {}
    '''calculate Macro scores'''
    num_samples = cm.shape[0]
    precision_list = np.zeros((num_samples))
    recall_list = np.zeros((num_samples))
    f1_list = np.zeros((num_samples))
    accuracy_list = np.zeros((num_samples))
    for i in range(cm.shape[0]):
        sub_cm = cm[i]
        tn = sub_cm[0, 0]
        fp = sub_cm[0, 1]
        fn = sub_cm[1, 0]
        tp = sub_cm[1, 1]

        precision = tp / float(tp + fp + r)
        recall = tp / float(tp + fn + r)
        precision_list[i] = precision
        recall_list[i] = recall
        f1 = 2 * (precision * recall / float(precision + recall + r))
        f1_list[i] = f1
        accuracy = (tp + tn) / float(tp + tn + fp + fn)
        accuracy_list[i] = accuracy
#     print("Macro acc: ",np.mean(accuracy_list),"Macro_precision: ", np.mean(precision_list),
#           "Macro_recall: ", np.mean(recall_list),"Macro_f1: ", np.mean(f1_list))
    dict_macro['macro_acc'] = np.mean(accuracy_list)
    dict_macro['macro_precision'] =  np.mean(precision_list)
    dict_macro['macro_recall'] = np.mean(recall_list)
    dict_macro['macro_f1'] = np.mean(f1_list)
    '''calculate Micro scores'''
    global_tp = cm[:,1,1].sum()
    global_tn = cm[:,0,0].sum()
    global_fn = cm[:,1,0].sum()
    global_fp = cm[:,0,1].sum()
    micro_accurracy = (global_tp + global_tn)/(global_tp + global_tn + global_fp + global_fn)
    micro_precision = global_tp/(global_tp + global_fp + r)
    micro_recall = global_tp/(global_tp + global_fn + r)
    micro_f1 = 2 * (micro_precision * micro_recall / float(micro_precision + micro_recall + r))
#     print(micro_accurracy,micro_precision,micro_recall,micro_f1)
    dict_micro['micro_acc'] = micro_accurracy
    dict_micro['micro_precision'] = micro_precision
    dict_micro['micro_recall'] = micro_recall
    dict_micro['micro_f1'] = micro_f1
    #print(dict_micro["micro_acc"])
    return dict_macro,dict_micro

def train(model, train_data, val_data, tokenizer, learning_rate, epochs):
    print("Inside train function")
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)
    print(train, len(train))
    print(val, len(val))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=16)
    print("Created data loaders")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Cuda status: ", use_cuda, device)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):
            print("Epoch: ", epoch_num)
            total_acc_train = 0
            total_loss_train = 0
            train_it, val_it = 0, 0
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device, dtype=float)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                print(type(output))
                
                batch_loss = criterion(output, train_label)

                total_loss_train += batch_loss.item()
                
                #acc = (output.argmax(dim=1) == train_label).sum().item()
                threshold = 0.5
                binary_prediction = output.cpu().detach().numpy().astype(np.int_)
                binary_prediction[ binary_prediction <= threshold ] = 0
                binary_prediction[ binary_prediction > threshold ] = 1
                dict_macro,dict_micro = compute_accuracy(train_label.cpu().detach().numpy().astype(np.int_), binary_prediction)
                acc = dict_micro["micro_acc"] 
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_it += 1
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device, dtype=float)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                   
                    threshold = 0.5
                    binary_prediction = output.cpu().detach().numpy().astype(np.int_)
                    binary_prediction[ binary_prediction <= threshold ] = 0
                    binary_prediction[ binary_prediction > threshold ] = 1
                    #acc = (output.argmax(dim=1) == val_label).sum().item()
                    dict_macro, dict_micro = compute_accuracy(val_label.cpu().detach().numpy().astype(np.int_), binary_prediction)
                    acc = dict_micro["micro_acc"]
                    total_acc_val += acc
                    val_it += 1
            #total_acc_train = total_acc_val = 0.0
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / train_it: .3f} | Train Accuracy: {total_acc_train / train_it: .3f} | Val Loss: {total_loss_val / val_it: .3f} | Val Accuracy: {total_acc_val / val_it: .3f}')
                  
def evaluate(model, test_data, tokenizer):

    test = Dataset(test_data, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

if __name__ == "__main__":
    np.random.seed(112)
    parser = argparse.ArgumentParser(description='TFT-Topaz command line arguments',\
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default=os.path.expanduser('./domains.csv'), help='input csv file')
    parser.add_argument('--model', default='bert-base-cased', help='hugginface model')
    parser.add_argument('--emb-size', default=768, type=int, help='hugginface model')
    parser.add_argument('--seq-len', default=512, type=int, help='hugginface model')
    args = parser.parse_args()

    datapath = args.input
    HFmodel = args.model
    emb_size = args.emb_size
    seq_len = args.seq_len
    df = pd.read_csv(datapath)
    if 'bert' in HFmodel:
        tokenizer = BertTokenizer.from_pretrained(HFmodel)
    elif 'gpt' in HFmodel:
        tokenizer = GPT2Tokenizer.from_pretrained(HFmodel)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
   
    print("train")
    df_train, uid1 = prepare_data("../data", "train_full.csv")
    print("test")
    df_val, uid2 = prepare_data("../data", "test_full.csv", uid1)
    print("dev")
    df_dev, uid3 = prepare_data("../data", "dev_full.csv", uid1)

    unique_labels = uid2
    EPOCHS = 1
    LR = 1e-6
    if 'bert' in HFmodel:
        model = BertClassifier(HFmodel, emb_size, nclasses=len(unique_labels))
    elif 'gpt' in HFmodel:
        model = GptClassifier(HFmodel, emb_size, seq_len, nclasses=len(unique_labels))
   
                  
    train(model, df_train, df_val, tokenizer, LR, EPOCHS)
    
    #evaluate(model, df_test, tokenizer)
