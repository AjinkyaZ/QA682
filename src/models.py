import torch
from torch import nn, optim
from torch.autograd import Variable as var
from torch.nn import functional as F
from utils import setup_glove

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
        if self.opt == 'Adam':
            self.opt = optim.Adam
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
        return var(weight.new(self.n_layers*self.dirs, bs, self.hidden_size).zero_())
        
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
        
        end_pred = lstm_op[:, -1, :self.hidden_size] # forward direction
        start_pred = lstm_op[:, -1, self.hidden_size:] # reverse direction
        
        # print("lstm start, end preds:", start_pred.size(), end_pred.size())
        out_start = F.log_softmax(self.decoder_start(start_pred), dim=-1)
        out_end = F.log_softmax(self.decoder_end(end_pred), dim=-1)
        # print("outs:", out_start.size(), out_end.size())
        out = torch.cat((out_start, out_end), 1)
        # print("out:", out.size())
        return out

    def fit(self, X, y):
        opt = self.opt(self.parameters(), self.lr)
        losses = [] 
        for epoch in range(self.epochs):
            print("epoch:", epoch)
            bs = self.batch_size
            bloss = 0.0 # epoch loss
            for bindex,  i in enumerate(range(0, len(y)-bs+1, bs)):
                #print("batch:", bindex)                    
                h, c = self.init_hidden(), self.init_hidden()
                self.hidden = (h, c)
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
                pred_spandiff = pred_span[:, 1] - pred_span[:, 0]
                
                loss = F.nll_loss(pred[:, :self.output_size], yb[:, 0]) \
                     + F.nll_loss(pred[:, self.output_size:], yb[:, 1])
                bloss += loss.data[0] # batch loss
                print(bindex, end=', ')
                loss.backward()
                opt.step()
            bloss /= bs
            losses.append(bloss)
            print("\nloss (epoch):", losses[-1], end=', change: ')
            if len(losses)>1:
                diff = losses[-2]-losses[-1]
                rel_diff = diff/losses[-2]
                print("%s"%rel_diff, "%")
            else:
                print("00.0%")
        return losses

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
        if self.opt == 'Adam':
            self.opt = optim.Adam
        elif self.opt == 'Adamax':
            self.opt = optim.Adamax
        else:
            self.opt = optim.SGD
        
        self.encoder_c = nn.Embedding(self.vocab_size, self.emb_dim)
        self.encoder_q = nn.Embedding(self.vocab_size, self.emb_dim)
        self.gru_c = nn.GRU(self.emb_dim, self.hidden_size, self.n_layers, bidirectional=self.bidir)
        self.gru_q = nn.GRU(self.emb_dim, self.hidden_size, self.n_layers, bidirectional=self.bidir)
        self.gru_op = nn.GRU(self.q_size*2, self.hidden_size, self.n_layers, bidirectional=self.bidir)
        self.decoder_start = nn.Linear(self.hidden_size, self.output_size)
        self.decoder_end = nn.Linear(self.hidden_size, self.output_size)
        self.init_weights()
    
    def init_weights(self):
        weight_scale = 0.01
        self.encoder_c.weight.data = self.vocab_weights
        self.encoder_q.weight.data = self.vocab_weights
        self.decoder_start.bias.data.fill_(0)
        self.decoder_start.weight.data.uniform_(-weight_scale, weight_scale)
        self.decoder_end.bias.data.fill_(0)
        self.decoder_end.weight.data.uniform_(-weight_scale, weight_scale)

    def init_hidden(self, bs=None):
        if bs is None:
            bs = self.batch_size
        weight = next(self.parameters()).data
        return var(weight.new(self.n_layers*self.dirs, bs, self.hidden_size).zero_())
        
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
        # print("embeds_c:", embeds_c.size())
        embeds_q = self.encoder_q(inputs_q).permute(1, 0, 2) # get glove repr
        # print("embeds_q:", embeds_q.size())

        c_op, self.hidden_c = self.gru_c(embeds_c, self.hidden_c)
        # print("c_op:", c_op.size())
        q_op, self.hidden_q = self.gru_q(embeds_q, self.hidden_q)
        # print("q_op:", q_op.size())
        # for bmm: c_op : (L1, N, H)->(N, L1, H), q_op : (L2, N, H) -> (N, H, L2)
        cq_op = torch.bmm(c_op.permute(1, 0, 2), q_op.permute(1, 2, 0))
        # print("cq_op:", cq_op.size())
        att_c = F.softmax(cq_op, dim=0)
        att_q = F.softmax(cq_op.permute(0, 2, 1), dim=0) # transpose while keeping batch
        # print("att_c:", att_c.size(), "att_q:", att_q.size())
        contx_q = torch.bmm(att_q, c_op.permute(1, 0, 2))
        # print("contx_q:", contx_q.size())
        q_contx_q = torch.cat((q_op, contx_q.permute(1, 0, 2)))
        # print("q_contx_q:", q_contx_q.size())
        # print(c_op.permute(1, 0, 2).size(), q_contx_q.permute(1, 2, 0).size())
        contx_c = torch.bmm(c_op.permute(1, 0, 2), q_contx_q.permute(1, 2, 0))
        # print("contx_c:", contx_c.size())
        gru_op, self.hidden_op = self.gru_op(contx_c.permute(1, 0, 2), self.hidden_op)
        gru_op = gru_op.permute(1, 0, 2) # revert to (N, L2, Hs# )
        # print("gru_op:", gru_op.size())
        end_pred = gru_op[:, -1, :self.hidden_size] # forward direction
        start_pred = gru_op[:, -1, self.hidden_size:] # reverse direction
        
        # print("gru start, end preds:", start_pred.size(), end_pred.size())
        out_start = F.log_softmax(self.decoder_start(start_pred), dim=-1)
        out_end = F.log_softmax(self.decoder_end(end_pred), dim=-1)
        # print("outs:", out_start.size(), out_end.size())
        out = torch.cat((out_start, out_end), 1)
        # print("out:", out.size())

        return out

    def fit(self, X, y):
        opt = self.opt(self.parameters(), self.lr)
        losses = [] # epoch loss
        for epoch in range(self.epochs):
            print("epoch:", epoch)
            bs = self.batch_size
            loss = 0.0
            for bindex,  i in enumerate(range(0, len(y)-bs+1, bs)):
                #print("batch:", bindex)                    
                self.hidden_c, self.hidden_q = self.init_hidden(), self.init_hidden()
                self.hidden_op = self.init_hidden()
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
                pred_spandiff = pred_span[:, 1] - pred_span[:, 0]
                
                bloss = F.nll_loss(pred[:, :self.output_size], yb[:, 0]) \
                     + F.nll_loss(pred[:, self.output_size:], yb[:, 1])
                loss += bloss.data[0]
                print(bindex, ':', round(bloss.data[0], 4))
                bloss.backward()
                opt.step()
            loss /= bs
            losses.append(loss)
            print("\nloss (epoch):", losses[-1], end=', change: ')
            if len(losses)>1:
                diff = losses[-2]-losses[-1]
                rel_diff = diff/losses[-2]
                print("%s"%rel_diff, "%")
            else:
                print("00.0%")
        return losses