import torch
from torch import nn, optim
from torch.autograd import Variable as var
from torch.nn import functional as F
from utils import setup_glove
from evaluate import exact_match_score, f1_score
from time import time
from pprint import pprint

class ModelV1(nn.Module):
    """
    LSTM Model - joint encoding of context and question  
    """
    def __init__(self, config):
        super(ModelV1, self).__init__()        
        
        self.input_size = config.get("input_size", 700)
        self.hidden_size = config.get("hidden_size", 128)
        self.output_size = config.get("output_size", 600)
        self.n_layers = config.get("n_layers", 1)
        self.vocab_weights = config.get("vocab", setup_glove().vectors)
        self.bidir = config.get("Bidirectional", True)
        self.dirs = int(self.bidir)+1
        self.lr = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 1)
        self.epochs = config.get("epochs", 5)
        self.opt = config.get("opt", "SGD")
        self.print_every = config.get("print_every", 10)
        
        self.vocab_size, self.emb_dim = self.vocab_weights.size()
        if self.opt == 'RMSProp':
            self.opt = optim.RMSProp
        elif self.opt == 'Adam':
            self.opt = optim.Adam
        elif self.opt == 'Adamax':
            self.opt = optim.Adamax
        elif self.opt == 'AdaDelta':
            self.opt = optim.Adadelta
        else:
            self.opt = optim.SGD

        self.encoder = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_size, self.n_layers, bidirectional=self.bidir)
        self.decoder_start = nn.Linear(self.hidden_size, self.output_size)
        self.decoder_end = nn.Linear(self.hidden_size, self.output_size)
        self.init_weights()
    
    def init_weights(self):
        weight_scale = 0.01
        self.encoder.weight.data = self.vocab_weights
        self.decoder_start.bias.data.fill_(0)
        self.decoder_start.weight.data.uniform_(-weight_scale, weight_scale)
        self.decoder_end.bias.data.fill_(0)
        self.decoder_end.weight.data.uniform_(-weight_scale, weight_scale)

    def init_hidden(self, bs=None):
        if bs is None:
            bs = self.batch_size
        weight = next(self.parameters()).data
        return var(weight.new(self.n_layers*self.dirs, bs, self.hidden_size)).zero_()

    def init_params(self, bs=None):
        h, c = self.init_hidden(bs), self.init_hidden(bs)
        self.hidden = (h, c)

        
    def forward(self, inputs):
        if len(inputs)==1:
            inputs = var(torch.LongTensor(inputs[0]))
        else:
            inputs = var(torch.LongTensor(inputs))
        if len(inputs.size())<2:
            inputs = inputs.unsqueeze(0)
        embeds = self.encoder(inputs).permute(1, 0, 2) # get glove repr
        # print("embeds:", embeds.size())
        seq_len = embeds.size()[0]
        lstm_op, self.hidden = self.lstm(embeds, self.hidden)
        # print("lstm op:", lstm_op.size()) # (seq_len, bs, hidden_size*(dirs=2 for bi))
        lstm_op = lstm_op.permute(1, 0, 2) # (seq_len, bs, hdim)->(bs, seq_len, hdim)
        
        end_pred = lstm_op[:, 0, :self.hidden_size] # forward direction
        start_pred = lstm_op[:, 0, self.hidden_size:] # reverse direction
        
        # print("lstm start, end preds:", start_pred.size(), end_pred.size())
        out_start = F.log_softmax(self.decoder_start(start_pred), dim=-1)
        out_end = F.log_softmax(self.decoder_end(end_pred), dim=-1)
        # print("outs:", out_start.size(), out_end.size())
        out = torch.cat((out_start, out_end), -1)
        if len(out.size())<2:
            out = out.unsqueeze(0)
        # print("out:", out.size())
        return out

    def fit(self, train_data, val_data):
        X, y = train_data
        X_val, y_val = val_data
        opt = self.opt(self.parameters(), self.lr)
        self.losses = [] # epoch loss
        self.val_losses = []
        bs = self.batch_size

        print("batch_size:", bs)
        print("batches:", len(X)/bs)
        for epoch in range(self.epochs):
            tic = time()
            print("epoch:", epoch)
            loss = 0.0
            for bindex,  i in enumerate(range(0, len(y)-bs+1, bs)):
                #print("batch:", bindex) 
                self.init_params(bs)
                # print(h.size(), c.size())
                opt.zero_grad()
                Xb = X[i:i+bs]
                Xb = torch.LongTensor(Xb)
                # print("Xb:", Xb.size())
                yb = var(torch.LongTensor(y[i:i+bs]))
                # print("yb:", yb.size())
                pred = self.forward(Xb) #prediction on batch features
                
                yb_spandiff = yb[:, 1] - yb[:, 0]
                pred_span = self.get_span_indices(pred)
                # print(pred_span)
                pred_spandiff = pred_span[:, 1] - pred_span[:, 0]
                
                bloss = F.nll_loss(pred[:, :self.output_size], yb[:, 0]) \
                      + F.nll_loss(pred[:, self.output_size:], yb[:, 1])
                loss += bloss.item()/bs
                print(bindex, ':', bloss.item())
                bloss.backward()
                opt.step()
            toc = time()-tic
            loss /= (len(y)/bs)
            self.losses.append(loss)
            print("\nloss (epoch):", self.losses[-1], end=', change: ')
            if len(self.losses)>1:
                diff = self.losses[-2]-self.losses[-1]
                rel_diff = diff/self.losses[-2]
                print("%s"%rel_diff, "%")
            else:
                print("00.0%", end=", took: %s seconds\n"%round(toc, 3))
            vloss = 0.0
            for bindex,  i in enumerate(range(0, len(y_val)-bs+1, bs)):
                X_valb = torch.LongTensor(X_val[i:i+bs])
                y_valb = var(torch.LongTensor(y_val[i:i+bs]))
                val_preds = self.forward(X_valb)
                vloss += (F.nll_loss(val_preds[:, :self.output_size], y_valb[:, 0]) \
                      + F.nll_loss(val_preds[:, self.output_size:], y_valb[:, 1])).item()
            vloss /= len(y_val)
            self.val_losses.append(vloss)
            print("validation loss:", vloss)

        return val_preds, self.losses, self.val_losses

    def predict(self, X, bs=None):
        # self.hidden = (self.init_hidden(bs), self.init_hidden(bs))
        result = self.forward(X)
        return self.get_span_indices(result)
    
    def get_span_indices(self, preds):
        s_pred = preds[:, :self.output_size]
        e_pred = preds[:, self.output_size:]
        _,  s_index = torch.max(s_pred, -1)
        _,  e_index = torch.max(e_pred, -1)
        return torch.cat((s_index.unsqueeze(1), e_index.unsqueeze(1)), -1)

class ModelV2(ModelV1):
    """
    Coattention with Answer Pointer
    """
    def __init__(self, config):
        super(ModelV1, self).__init__()        
        
        self.c_size = config.get("context_size", 600)
        self.q_size = config.get("question_size", 100)
        self.hidden_size = config.get("hidden_size", 128)
        self.output_size = config.get("output_size", 600)
        self.n_layers = config.get("n_layers", 1)
        self.vocab_weights = config.get("vocab", setup_glove().vectors)
        self.bidir = config.get("Bidirectional", True)
        self.dirs = int(self.bidir)+1
        self.lr = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 1)
        self.epochs = config.get("epochs", 5)
        self.opt = config.get("opt", "SGD")
        self.print_every = config.get("print_every", 10)
        self.vocab_size, self.emb_dim = self.vocab_weights.size()
        if self.opt == 'RMSProp':
            self.opt = optim.RMSProp
        elif self.opt == 'Adam':
            self.opt = optim.Adam
        elif self.opt == 'Adamax':
            self.opt = optim.Adamax
        elif self.opt == 'AdaDelta':
            self.opt = optim.Adadelta
        else:
            self.opt = optim.SGD
        
        self.encoder_c = nn.Embedding(self.vocab_size, self.emb_dim)
        self.encoder_q = nn.Embedding(self.vocab_size, self.emb_dim)

        self.gru_c = nn.GRU(self.emb_dim, self.hidden_size, self.n_layers)
        self.gru_q = nn.GRU(self.emb_dim, self.hidden_size, self.n_layers)

        self.lin_q = nn.Linear(self.hidden_size, self.hidden_size)

        self.gru_coatt = nn.GRU(3*self.hidden_size, 2*self.dirs*self.hidden_size, self.n_layers, bidirectional=self.bidir)

        self.gru_bmod = nn.GRU(4*self.dirs*self.hidden_size, self.hidden_size, self.n_layers)

        self.ans_ptr_1 = nn.Linear(4*self.hidden_size*self.dirs, self.hidden_size) #V
        self.ans_ptr_2 = nn.Linear(self.hidden_size, self.hidden_size) #W
        self.ans_ptr_3 = nn.Linear(self.hidden_size, 1) #v

        self.decoder_start = nn.Linear(self.hidden_size*2*self.dirs, self.output_size)
        self.decoder_end = nn.Linear(self.hidden_size*2*self.dirs, self.output_size)
        self.init_weights()
    
    def init_weights(self):
        weight_scale = 0.01
        self.encoder_c.weight.data = self.vocab_weights
        self.encoder_q.weight.data = self.vocab_weights
        
        self.lin_q.bias.data.fill_(0)
        self.lin_q.weight.data.uniform_(-weight_scale, weight_scale)
        
        self.ans_ptr_1.bias.data.fill_(0)
        self.ans_ptr_1.weight.data.uniform_(-weight_scale, weight_scale)
        self.ans_ptr_2.bias.data.fill_(0)
        self.ans_ptr_2.weight.data.uniform_(-weight_scale, weight_scale)
        self.ans_ptr_3.bias.data.fill_(0)
        self.ans_ptr_3.weight.data.uniform_(-weight_scale, weight_scale)

        self.decoder_start.bias.data.fill_(0)
        self.decoder_start.weight.data.uniform_(-weight_scale, weight_scale)
        self.decoder_end.bias.data.fill_(0)
        self.decoder_end.weight.data.uniform_(-weight_scale, weight_scale)

    def init_hidden(self, bs=None):
        weight_scale = 0.01
        if bs is None:
            bs = self.batch_size
        weight = next(self.parameters()).data
        return var(weight.new(self.n_layers, bs, self.hidden_size)).zero_()

    def init_hidden_coatt(self, bs=None): # for context_attention GRU
        weight_scale = 0.01
        if bs is None:
            bs = self.batch_size
        weight = next(self.parameters()).data
        return var(weight.new(self.n_layers*self.dirs, bs, self.hidden_size*2*self.dirs)).zero_()

    def init_hidden_bmod(self, bs=None):
        weight_scale = 0.01
        if bs is None:
            bs = self.batch_size
        weight = next(self.parameters()).data
        return var(weight.new(self.n_layers, bs, self.hidden_size)).uniform_(-weight_scale, weight_scale)
 
    def init_params(self, bs=None):
        self.hidden_c, self.hidden_q = self.init_hidden(bs), self.init_hidden(bs)
        self.hidden_coatt = self.init_hidden_coatt(bs)
        self.hidden_bmod1 = self.init_hidden_bmod(bs)
        self.beta = var(torch.ones((bs, self.c_size,))*(1/self.c_size))

    def forward(self, inputs):
        if len(inputs)==1:
            inputs = var(torch.LongTensor(inputs[0]))
        else:
            inputs = var(torch.LongTensor(inputs))
        if len(inputs.size())<2:
            inputs = inputs.unsqueeze(0)

        inputs_c = inputs[:, :self.c_size]
        inputs_q = inputs[:, self.c_size:self.c_size+self.q_size]
        embeds_c = self.encoder_c(inputs_c).permute(1, 0, 2) # get glove repr
        embeds_q = self.encoder_q(inputs_q).permute(1, 0, 2) # get glove repr

        c_op, self.hidden_c = self.gru_c(embeds_c, self.hidden_c)
        q_op, self.hidden_q = self.gru_q(embeds_q, self.hidden_q)

        # coattention network

        q_op = F.tanh(self.lin_q(q_op))
        cq_op = torch.bmm(c_op.permute(1, 0, 2), q_op.permute(1, 2, 0))
        att_q = F.softmax(cq_op, dim=0)
        att_c = F.softmax(cq_op.permute(0, 2, 1), dim=0) # transpose while keeping batch
        contx_q = torch.bmm(att_q.permute(0,2,1), c_op.permute(1, 0, 2))
        q_contx_q = torch.cat((q_op, contx_q.permute(1, 0, 2)), 2)
        contx_c = torch.bmm(q_contx_q.permute(1, 2, 0), att_c)
        c_contx_c = torch.cat((contx_c.permute(0, 2, 1), c_op.permute(1, 0, 2)), 2)
        coatt_op, self.hidden_coatt = self.gru_coatt(c_contx_c.permute(1, 0, 2), self.hidden_coatt)
        coatt_op = coatt_op.permute(1, 0, 2) # revert to (N, L2, Hs# )


        # answer pointer - boundary model:

        f1 = self.ans_ptr_1(coatt_op)
        H_beta = torch.bmm(self.beta.unsqueeze(1), coatt_op)
        W_h = self.ans_ptr_2(self.hidden_bmod1).repeat(self.c_size, 1, 1)
        f1 += W_h.permute(1, 0, 2)
        f1 = F.tanh(f1)
        out_start = F.log_softmax(self.ans_ptr_3(f1), 0).squeeze()

        f2 = self.ans_ptr_1(coatt_op)
        bmod_2, self.hidden_bmod2 = self.gru_bmod(H_beta.permute(1, 0, 2), self.hidden_bmod1)
        W_h = self.ans_ptr_2(self.hidden_bmod2).repeat(self.c_size, 1, 1)
        f2 += W_h.permute(1, 0, 2)
        f2 = F.tanh(f2)
        out_end = F.log_softmax(self.ans_ptr_3(f2), 0).squeeze()

        out = torch.cat((out_start, out_end), -1)
        if len(out.size())<2:
            out = out.unsqueeze(0)
        # print("out:", out.size())

        return out

    def fit(self, train_data, val_data):
        X, y = train_data
        X_val, y_val = val_data
        opt = self.opt(self.parameters(), self.lr)
        self.losses = [] # epoch loss
        self.val_losses = []
        bs = self.batch_size

        print("batch_size:", bs)
        print("batches:", len(X)/bs)
        for epoch in range(self.epochs):
            tic = time()
            print("epoch:", epoch)
            loss = 0.0
            for bindex,  i in enumerate(range(0, len(y)-bs+1, bs)):
                #print("batch:", bindex)
                self.init_params(bs)               
                opt.zero_grad()
                Xb = X[i:i+bs]
                Xb = torch.LongTensor(Xb)
                # print("Xb:", Xb.size())
                yb = var(torch.LongTensor(y[i:i+bs]))
                # print("yb:", yb.size())
                pred = self.forward(Xb) #prediction on batch features
                
                yb_spandiff = yb[:, 1] - yb[:, 0]
                """
                pred_span = self.get_span_indices(pred)
                # print(pred_span)
                pred_spandiff = pred_span[:, 1] - pred_span[:, 0]
                """
                bloss = F.nll_loss(pred[:, :self.output_size], yb[:, 0]) \
                      + F.nll_loss(pred[:, self.output_size:], yb[:, 1])
                loss += bloss.item()/bs
                print(bindex, ':', bloss.item()/bs)
                bloss.backward()
                opt.step()
            toc = time()-tic
            loss /= (len(y)/bs)
            self.losses.append(loss)
            print("\nloss (epoch):", self.losses[-1], end=', change: ')
            if len(self.losses)>1:
                diff = self.losses[-2]-self.losses[-1]
                rel_diff = diff/self.losses[-2]
                print("%s"%rel_diff, "%")
            else:
                print("00.0%", end=", took: %s seconds\n"%round(toc, 3))
            vloss = 0.0
            for bindex,  i in enumerate(range(0, len(y_val)-bs+1, bs)):
                X_valb = torch.LongTensor(X_val[i:i+bs])
                y_valb = var(torch.LongTensor(y_val[i:i+bs]))
                val_preds = self.forward(X_valb)
                vloss += (F.nll_loss(val_preds[:, :self.output_size], y_valb[:, 0]) \
                      + F.nll_loss(val_preds[:, self.output_size:], y_valb[:, 1])).item()
            vloss /= len(y_val)
            self.val_losses.append(vloss)
            print("validation loss:", vloss)

        return val_preds, self.losses, self.val_losses