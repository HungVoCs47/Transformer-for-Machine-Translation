import torch
import spacy
import torchtext
from torchtext.legacy import data
import numpy as np
import torch
import torch.nn.functional as F
from Transformer import Transformer
import time
import numpy as np
from tqdm.auto import tqdm

d_model = 512
n_heads = 8
N = 6
src_vocab_size = 13756
trg_vocab_size = 26341  
model = Transformer(src_vocab_size, trg_vocab_size, d_model, N, n_heads)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)    

for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform(p) 

def create_mask(src_input, trg_input, SRC:torchtext.legacy.data.field.Field):

    pad = SRC.vocab.stoi['<pad>']
    src_mask = (src_input != pad).unsqueeze(1)
    trg_mask = (trg_input != pad).unsqueeze(1)
    seq_len = trg_input.size(1)
    nopeak_mask = np.tril(np.ones((1, seq_len, seq_len)), k=0).astype('uint8')
    nopeak_mask = torch.from_numpy(nopeak_mask) != 0
    trg_mask = trg_mask & nopeak_mask
    return src_mask, trg_mask

def train_model(n_epochs, SRC:torchtext.legacy.data.field.Field, train_iter:torchtext.legacy.data.iterator.Iterator ,output_interval=100):
    
    model.train()
    start = time.time() 
    
    for epoch in tqdm(range(n_epochs)):
        log_dir = 'pretrain/'
        total_loss = 0
        for i, batch in enumerate(train_iter):
            
            src_input = batch.src.transpose(0, 1) 
            #(src_input) 
            trg = batch.trg.transpose(0, 1)  
            #print(trg)
            trg_input = trg[:, :-1]
            ys = trg[:, 1:].contiguous().view(-1)
            
            
            src_mask, trg_mask = create_mask(src_input, trg_input, SRC = SRC)
            preds = model(src_input, trg_input, src_mask, trg_mask)
            
            optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=1)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
           
            if (i + 1) % output_interval == 0:
                avg_loss = total_loss/output_interval
                time_ = (time.time() - start)/60
                my_formatter = "{0:.2f}"
                print('time = {}, epoch = {}, iter = {}, loss = {}'.format(my_formatter.format(time_),
                                                                           epoch + 1,
                                                                           i + 1,
                                                                           my_formatter.format(avg_loss)))
                total_loss = 0
                start = time.time()
                log_msg = str('time = {}, epoch = {}, iter = {}, loss = {}'.format(my_formatter.format(time_),
                                                                           epoch + 1,
                                                                           i + 1,
                                                                           my_formatter.format(avg_loss)))
                
                with open(f'{log_dir}/train.log', 'a') as f:
                    f.write(log_msg + '\n')
                
        ckpt = model.state_dict()
            
        torch.save(ckpt, f=f"{log_dir}/Transformer_{epoch}.pt")
                

