#from transformers import BartTokenizer, BartForConditionalGeneration
# from modeling_bart import BartForConditionalGeneration
from transformers import BertTokenizer, BertForMaskedLM

import os
import pdb
import random
import torch 
from torch import nn
import torchvision.models as models
import argparse
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def list_to_str(list):
    item_all=""
    for item in list:
        item = item+" "
        item_all=item_all+item
    return item_all[:-1]

#mask所有有标签的词
def ner_mask(text,label):
    for i in range (0,len(label)):
        if label[i]!="O":
            text[i]="[MASK]"
    return text, label

#返回mask原文, mask所有有标签的词
def ner_preprocess(file_path: str,file_path2: str): 
    token_file_path = file_path
    label_file_path = file_path2
    with open(token_file_path, "r", encoding="utf-8") as f_token:
        token_lines = f_token.readlines()
    with open(label_file_path, "r", encoding="utf-8") as f_label:
        label_lines = f_label.readlines()
    # assert len(token_lines) == len(label_lines)
    token_all=[]
    # import pdb
    # pdb.set_trace()
    for token_line, label_line in zip(token_lines, label_lines):
        if not token_line.strip() or not label_line.strip():
            continue
        tokens = token_line.strip().split(" ")
        labels = label_line.strip().split(" ")
       
        # import pdb
        # pdb.set_trace()
        assert len(tokens)==len(labels),"输入与标签存在不相等"
        
        tokens,labels=ner_mask(tokens,labels)
        
        if len(tokens) == 0 or len(labels) == 0:
            continue
        tokens = [token.strip() for token in tokens if token.strip()]
        
        
        tokens=list_to_str(tokens)  #转化成字符串进行append
        
        # import pdb
        # pdb.set_trace()
        
        token_all.append(tokens)
        
    return token_all 

#返回未处理的原文
def ner_origin_text(file_path: str,file_path2: str):    ##读取原文
    '''返回原文'''
    token_file_path = file_path
    label_file_path = file_path2
    with open(token_file_path, "r", encoding="utf-8") as f_token:
        token_lines = f_token.readlines()
    with open(label_file_path, "r", encoding="utf-8") as f_label:
        label_lines = f_label.readlines()
    # assert len(token_lines) == len(label_lines)
    token_all=[]
    label_all=[]
    origin_all=[]
    # import pdb
    # pdb.set_trace()
    for token_line, label_line in zip(token_lines, label_lines):
        if not token_line.strip() or not label_line.strip():
            continue
        tokens = token_line.strip().split(" ")
        labels = label_line.strip().split(" ")
        origin_tokens=tokens   #加载了原文本
        origin_all.append(origin_tokens)
        
        
        if len(tokens) == 0 or len(labels) == 0:
            continue
        tokens = [token.strip() for token in tokens if token.strip()]
        labels = [label.strip() for label in labels if label.strip()]
        
        tokens=list_to_str(tokens)  #转化成字符串进行append
        labels=list_to_str(labels)
        # import pdb
        # pdb.set_trace()
        
        token_all.append(tokens)
        label_all.append(labels)
    
    return token_all






def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help="train mode")
    parser.add_argument('--epoch', type=int, default=5, help='epoch num')
    parser.add_argument('--train_data', default='/aa/bb/cc/inter_train_10_5_seq_in.txt', help='train_data dir')
    parser.add_argument('--train_data2', default='/aa/bb/cc/inter_train_10_5_seq_out.txt', help='train_data dir')
    parser.add_argument('--model_save_dir', default='./dd/', help='train_data dir')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='crop size of image')
    parser.add_argument('--batch_size', type=int, default=4, help='epoch num')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--model_path', default='/aa/bb/cc/dd/intra10-5/intra_train_10_5_model_4.bin', help='random seed')
    

    args = parser.parse_args()
    random.seed(args.seed)
    
    if args.do_train:
        print("训练模式")
    #if args.test_only:
    #    print("测试模式")
    
    
    print("loading train & test data")
    
    
    train_tokens = ner_preprocess(args.train_data,args.train_data2)
    train_origin_tokens = ner_origin_text(args.train_data,args.train_data2)
    
    print("finish loading")

    print("tokenizer loading")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("finish loading")
    
    # for i,text in enumerate(tqdm(train_origin_tokens[:100])):
    #     train_ids = tokenizer(text, return_tensors="pt",padding="max_length", max_length=max(len(s) for s in train_origin_tokens) )["input_ids"].to(device) #原句（不mask）的ids
    #     train_origin_ids.append(train_ids)
    # # import pdb
    # # pdb.set_trace()    
    # print("finish train_origin_ids loading")
    
    # train_mask_ids=[]
    # for i,text in enumerate(tqdm(train_tokens[:100])):
    #     mask_ids = tokenizer(text, return_tensors="pt",padding="max_length", max_length=max(len(s) for s in train_origin_tokens))["input_ids"].to(device) #增强句子的ids  
    #     train_mask_ids.append(mask_ids)
    # print("finish train_mask_ids loading")    
    
   
    # train_origin_ids,train_mask_ids=torch.stack(train_origin_ids),torch.stack(train_mask_ids)
    #tokenizer(train_origin_tokens, return_tensors="pt",padding=True)["input_ids"].to(device) 
    
    train_origin_ids = tokenizer(train_origin_tokens, return_tensors="pt",padding="max_length",max_length=350,truncation=True)["input_ids"].to(device) 
    print("train_origin_ids finish loading")
    train_mask_ids = tokenizer(train_tokens, return_tensors="pt",padding="max_length",max_length=350,truncation=True)["input_ids"].to(device)
    print("train_augment_ids finish loading")  
    
    train_ids = TensorDataset(train_mask_ids,train_origin_ids)   
    train_loader = DataLoader(dataset=train_ids, batch_size=args.batch_size, shuffle=False)
    
    
    print("data loader process success")
    print(f"batch_size:{args.batch_size}")

    
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")  
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    count=0
    
    
    learning_rate = args.learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    running_loss = 0.0
    if args.do_train:
        print("------------------------------")
        print("training begining!")
        print("------------------------------")
        model.train()
        for epoch in range(args.epoch):
            tq = tqdm(enumerate(train_loader, 0)) 
            for step, data in enumerate(train_loader, 0): #这里的data就是train_loader的每一个元组（mask的输入embedding，原文输入embedding）
                
                input_ids, origin_ids = data
                # import pdb
                # pdb.set_trace()
                #logits = model(input_ids).logits.to(device)   
                logits = model(input_ids).logits.to(device)
                
                loss = model(input_ids,labels=origin_ids).loss
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count+=1
                
                running_loss += loss.item()
                
                if step % 20 == 19:   #打印每2000步mini batch
                    tq.set_description('[epoch: %d, step: %5d] loss: %.3f' % (epoch + 1, count+1, running_loss / 20))
                    running_loss = 0.0
            count = 0
            # import pdb
            # pdb.set_trace()
            torch.save(model.state_dict(),os.path.join(args.model_save_dir,f'inter_train_5_1_model_{epoch}.bin'))
        print("finish training")
           
if __name__ == "__main__":
    main()