#from transformers import BartTokenizer, BartForConditionalGeneration
# from modeling_bart import BartForConditionalGeneration
from http.client import ImproperConnectionState
from turtle import pd
from transformers import BertTokenizer, BertForMaskedLM
from transformers.models.bert import BertModel
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

def ner_origin_text(file_path: str):    ##读取原文
    '''返回原文'''
    token_file_path = file_path
    with open(token_file_path, "r", encoding="utf-8") as f_token:
        token_lines = f_token.readlines()
    token_all=[]
    label_all=[]
    origin_all=[]
    for token_line in token_lines:
        tokens = token_line.strip().split(" ")
        origin_tokens=tokens   #加载了原文本
        origin_all.append(origin_tokens)
        tokens = [token.strip() for token in tokens if token.strip()]    
        tokens=list_to_str(tokens)  #转化成字符串进行append
        token_all.append(tokens)
    return token_all

def cos_similarity(mat1, mat2, temperature):
    norm1 = torch.norm(mat1, p=2, dim=1).view(-1, 1)
    norm2 = torch.norm(mat2, p=2, dim=1).view(1, -1)
    # import pdb
    # pdb.set_trace()
    cos_sim = torch.matmul(mat1, mat2.t()) / torch.matmul(norm1, norm2)
    norm_sim = torch.exp(cos_sim / temperature)
    return norm_sim

def info_NCE(sim_pos, sim_total):
    deno = torch.sum(sim_total) - sim_total[0][0] + sim_pos[0][0]
    loss = -torch.log(sim_pos[0][0] / deno)
    return loss

def instance_CL_Loss(ori_hidden, aug_hidden, type="origin", temp=0.5):
    inputs_mean = ori_hidden
    positive_examples_mean = aug_hidden
    batch_size = inputs_mean.size()[0]
    cons_loss = 0
    count = 0
    for ori_input, pos_exp in zip(inputs_mean, positive_examples_mean):
        ori_input = torch.reshape(ori_input, (1, -1))
        pos_exp = torch.reshape(pos_exp, (1, -1))
        # import pdb
        # pdb.set_trace()
        sim_pos = cos_similarity(ori_input, pos_exp, temp)
        '''魔改对比学习，让他去和其余的负样本对比'''
        # negative=torch.Tensor()
        if type == "origin":
            negative = positive_examples_mean[torch.arange(positive_examples_mean.size(0)) != count]  # batch中其他样本作为negative
        elif type == "aug":
            negative = positive_examples_mean[torch.arange(positive_examples_mean.size(0)) != count]  # 和其他batch样本的keyword 拉大
        count += 1
        # import pdb
        # pdb.set_trace()
        negative = torch.reshape(negative, (batch_size-1, -1))
        sim_total = cos_similarity(ori_input, negative, temp)  ##修改为了正样例的平均
        cur_loss = info_NCE(sim_pos, sim_total)
        cons_loss += cur_loss

    return cons_loss / batch_size
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help="train mode")
    parser.add_argument('--epoch', type=int, default=5, help='epoch num')
    parser.add_argument('--train_data', default='/aa/bb/cc/dd/train_5_1_origin_text.txt', help='train_data dir')
    parser.add_argument('--train_data2', default='/aa/bb/cc/dd/train_5_1_augment_text.txt', help='train_data dir')
    parser.add_argument('--model_save_dir', default='./ee/', help='train_data dir')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='model_save_dir')
    parser.add_argument('--batch_size', type=int, default=5, help='epoch num')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--temp', default='0.5', help='temperature')
    
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    if args.do_train:
        print("训练模式")
   

    print("loading train & test data")
   
    
    
    train_origin_tokens = ner_origin_text(args.train_data)
    train_augment_tokens = ner_origin_text(args.train_data2)
   
    print("finish loading")

    
    
    
    
    '''train_dataloader封装'''
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_origin_ids = tokenizer(train_origin_tokens, return_tensors="pt",padding="max_length",max_length=300,truncation=True)["input_ids"].to(device) 
    print("train_origin_ids finish loading")
    train_augment_ids = tokenizer(train_augment_tokens, return_tensors="pt",padding="max_length",max_length=300,truncation=True)["input_ids"].to(device)
    print("train_augment_ids finish loading")
    train_ids = TensorDataset(train_augment_ids,train_origin_ids)   
    train_loader = DataLoader(dataset=train_ids, batch_size=args.batch_size, shuffle=False)
    
    
 
    print("data loader process success")
    print(f"batch_size:{args.batch_size}")

    
    # TXT = "I want to go to the <mask> because i am hungry."
    #model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    
    #model = BertModel.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.to(device)
    
    # import pdb
    # pdb.set_trace()
    count=0
    
    learning_rate = args.learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    running_loss = 0.0
   
   
    '''training'''
    if args.do_train:
        print("------------------------------")
        print("training begining!")
        print("------------------------------")
        model.train()
        for epoch in range(args.epoch):
            tq = tqdm(enumerate(train_loader, 0)) 
            for step, data in enumerate(train_loader, 0):  #这里的data就是train_loader的每一个元组（mask的输入embedding，原文输入embedding）
                
                augment_ids, origin_ids = data
                # import pdb
                # pdb.set_trace()
                #logits = model(input_ids).logits.to(device)  
                 
                #origin_logits = model(origin_ids).last_hidden_state
                #augment_logits = model(augment_ids).last_hidden_state
                
                # origin_logits = model(origin_ids).pooler_output
                # augment_logits = model(augment_ids).pooler_output
                
                # import pdb
                # pdb.set_trace()
                
                origin_logits = model(origin_ids).logits.to(device)
                augment_logits = model(augment_ids).logits.to(device)
                # import pdb
                # pdb.set_trace()
                
                loss = instance_CL_Loss(origin_logits, augment_logits, 'origin', 0.5)
                #loss = model(origin_ids,labels=augment_ids).loss
                
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
            torch.save(model.state_dict(),os.path.join(args.model_save_dir,f'constra_inter_10_5_model_{epoch}.bin'))
            #torch.save(model,os.path.join(args.model_save_dir,f'model_{epoch}.pth'))
        print("finish training")
    

           
if __name__ == "__main__":
    main()