from pprint import pprint
from time import time

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
        self.conf = config
        self.input_size = config.get("input_size", 700)
        self.hidden_size = config.get("hidden_size", 64)
        self.output_size = config.get("output_size", 600)
        self.n_layers = config.get("n_layers", 1)
        self.vocab_weights = config.get("vocab", setup_glove().vectors)
        self.bidir = config.get("Bidirectional", True)
        self.dirs = int(self.bidir)+1
        self.lr = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 10)
        self.opt_name = config.get("opt", "Adam")
        self.save_every = config.get("save_every", 5)

        self.vocab_size, self.emb_dim = self.vocab_weights.size()
        if self.opt_name == 'RMSProp':
            self.opt = optim.RMSProp
        elif self.opt_name == 'Adam':
            self.opt = optim.Adam
        elif self.opt_name == 'Adamax':
            self.opt = optim.Adamax
        elif self.opt_name == 'AdaDelta':
            self.opt = optim.Adadelta
        else:
            self.opt = optim.SGD

        self.encoder = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_size,
                            self.n_layers, bidirectional=self.bidir)
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
        return var(weight.new(self.n_layers*self.dirs,
                              bs, self.hidden_size)).zero_()

    def init_params(self, bs=None):
        h, c = self.init_hidden(bs), self.init_hidden(bs)
        self.hidden = (h, c)

    def forward(self, inputs):
        if torch.cuda.is_available():
            if len(inputs) == 1:
                inputs = var(torch.cuda.LongTensor(inputs[0]))
            else:
                inputs = var(torch.cuda.LongTensor(inputs))
        else:
            if len(inputs) == 1:
                inputs = var(torch.LongTensor(inputs[0]))
            else:
                inputs = var(torch.LongTensor(inputs))
        embeds = self.encoder(inputs).permute(1, 0, 2)  # get glove repr
        # print("embeds:", embeds.size())
        seq_len = embeds.size()[0]
        lstm_op, self.hidden = self.lstm(embeds, self.hidden)
        # print("lstm op:", lstm_op.size())
        # (seq_len, bs, hidden_size*(dirs=2 for bi))
        # (seq_len, bs, hdim)->(bs, seq_len, hdim)
        lstm_op = lstm_op.permute(1, 0, 2)

        end_pred = lstm_op[:, 0, :self.hidden_size]  # forward direction
        start_pred = lstm_op[:, 0, self.hidden_size:]  # reverse direction

        # print("lstm start, end preds:", start_pred.size(), end_pred.size())
        out_start = F.log_softmax(self.decoder_start(start_pred), dim=-1)
        out_end = F.log_softmax(self.decoder_end(end_pred), dim=-1)
        # print("outs:", out_start.size(), out_end.size())
        out = torch.cat((out_start, out_end), -1)
        if len(out.size()) < 2:
            out = out.unsqueeze(0)
        # print("out:", out.size())
        return out

    def fit(self, train_data, val_data, pretrained=False):
        """
        pretrained is either False or a dict with params
        """
        X, y = train_data
        X_val, y_val = val_data

        opt_ = self.conf["opt"]
        self.name = f"{type(self).__name__}_D{len(X)}_B{self.batch_size}_" \
                     f"E{self.epochs}_H{self.hidden_size}_LR{self.lr}_O{opt_}"

        opt = self.opt(self.parameters(), self.lr)
        new_epochs = 0
        new_data_size = 0
        if pretrained:
            new_epochs = pretrained["epochs"]
            new_data_size = pretrained["new_data"]
        else:
            self.losses = []  # epoch loss
            self.val_losses = []
        bs = self.batch_size

        print(f"batch_size: {bs}")
        print(f"batches: {len(X)//bs}")

        nll_loss = nn.NLLLoss(ignore_index=400002)

        self.train()
        for epoch in range(self.epochs+new_epochs):
            tic = time()
            print(f"epoch {epoch}")
            loss = 0.0

            # set to train mode
            for bindex,  i in enumerate(range(0, len(y)-bs+1, bs)):
                if not pretrained:
                    self.init_params(bs)
                opt.zero_grad()
                Xb = X[i:i+bs]

                Xb = torch.LongTensor(Xb)
                yb = var(torch.LongTensor(y[i:i+bs]))
                if torch.cuda.is_available():
                    Xb = Xb.cuda()
                    yb = yb.cuda()

                pred = self.forward(Xb)  # prediction on batch features

                yb_spandiff = yb[:, 1] - yb[:, 0]

                bloss = nll_loss(pred[:, :self.output_size], yb[:, 0]) \
                    + nll_loss(pred[:, self.output_size:], yb[:, 1])
                loss += bloss.item()/bs

                print(f"batch {bindex} : {bloss.item()/bs:0.6f}")
                bloss.backward()
                opt.step()

            toc = time()-tic
            loss /= (len(y)/bs)
            self.losses.append(loss)
            print(f"\nloss (epoch): {self.losses[-1]:0.6f}",
                  end=", change: ")
            if len(self.losses) > 1:
                diff = self.losses[-2]-self.losses[-1]
                rel_diff = diff/self.losses[-2]
                print(f"{rel_diff*100:0.4f}",
                      end=f", took: {toc:0.3f} seconds\n")
            else:
                print("00.0%", end=f", took: {toc:0.3f} seconds\n")

            vloss = 0.0
            for bindex,  i in enumerate(range(0, len(y_val)-bs+1, bs)):
                X_valb = torch.LongTensor(X_val[i:i+bs])
                y_valb = var(torch.LongTensor(y_val[i:i+bs]))
                self.init_params(bs)
                if torch.cuda.is_available():
                    X_valb = X_valb.cuda()
                    y_valb = y_valb.cuda()
                val_preds = self.forward(X_valb)

                # add loss for start and end tokens
                start_loss = nll_loss(
                    val_preds[:, :self.output_size], y_valb[:, 0])
                end_loss = nll_loss(
                    val_preds[:, self.output_size:], y_valb[:, 1])
                vloss += (start_loss + end_loss)

            vloss /= len(y_val)
            self.val_losses.append(vloss.item())
            print("validation loss:", round(vloss.item(), 6))
            if epoch % self.save_every == 0:
                epoch_model_name = \
                    f"../evaluation/models/{self.name}_onEpoch_{epoch}"
                print(f"Saving model to {epoch_model_name}...", end="..")
                torch.save(self, epoch_model_name)
                print("Saved!")

        self.epochs += new_epochs
        self.name = f"{type(self).__name__}_D{len(X)+new_data_size}" \
            f"_B{self.batch_size}_E{self.epochs}_" \
            f"H{self.hidden_size}_LR{self.lr}_O{self.opt_name}"
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
        self.conf = config
        self.c_size = config.get("context_size", 600)
        self.q_size = config.get("question_size", 100)
        self.cell_type = config.get("cell_type", "GRU")
        self.hidden_size = config.get("hidden_size", 128)
        self.output_size = config.get("output_size", 600)
        self.n_layers = config.get("n_layers", 1)
        self.vocab_weights = config.get("vocab", setup_glove().vectors)
        self.bidir = config.get("Bidirectional", True)
        self.dirs = int(self.bidir)+1
        self.lr = config.get("learning_rate", 1e-3)
        self.linear_dropout = config.get("linear_dropout", 0.0)
        self.seq_dropout = config.get("seq_dropout", 0.0)
        self.batch_size = config.get("batch_size", 1)
        self.epochs = config.get("epochs", 5)
        self.opt_name = config.get("opt", "SGD")
        self.save_every = config.get("save_every", 5)

        self.vocab_size, self.emb_dim = self.vocab_weights.size()
        if self.opt_name == 'RMSProp':
            self.opt = optim.RMSProp
        elif self.opt_name == 'Adam':
            self.opt = optim.Adam
        elif self.opt_name == 'Adamax':
            self.opt = optim.Adamax
        elif self.opt_name == 'AdaDelta':
            self.opt = optim.Adadelta
        else:
            self.opt = optim.SGD

        if self.cell_type not in ["LSTM", "GRU"]:
            raise TypeError("Invalid Cell Type - use LSTM or GRU")
        self.cell_present = False
        self.encoder_c = nn.Embedding(self.vocab_size, self.emb_dim)
        self.encoder_q = nn.Embedding(self.vocab_size, self.emb_dim)

        self.gru_c = getattr(nn, self.cell_type)(
            self.emb_dim, self.hidden_size, self.n_layers)
        self.gru_q = getattr(nn, self.cell_type)(
            self.emb_dim, self.hidden_size, self.n_layers)

        self.lin_q = nn.Linear(self.hidden_size, self.hidden_size)

        self.gru_coatt = getattr(nn, self.cell_type)(3*self.hidden_size,
                                                     2*self.dirs *
                                                     self.hidden_size,
                                                     self.n_layers,
                                                     bidirectional=self.bidir,
                                                     dropout=self.seq_dropout)

        self.gru_bmod = getattr(nn, self.cell_type)(
            4*self.dirs*self.hidden_size, self.hidden_size, self.n_layers,
            dropout=self.seq_dropout)

        self.ans_ptr_1 = nn.Linear(
            4*self.hidden_size*self.dirs, self.hidden_size)  # V
        self.ans_ptr_2 = nn.Linear(self.hidden_size, self.hidden_size)  # W
        self.ans_ptr_3 = nn.Linear(self.hidden_size, 1)  # v

        self.decoder_start = nn.Linear(
            self.hidden_size*2*self.dirs, self.output_size)
        self.decoder_end = nn.Linear(
            self.hidden_size*2*self.dirs, self.output_size)
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

    def init_hidden_coatt(self, bs=None):  # for context_attention GRU
        weight_scale = 0.01
        if bs is None:
            bs = self.batch_size
        weight = next(self.parameters()).data
        return var(weight.new(self.n_layers*self.dirs, bs,
                              self.hidden_size*2*self.dirs)).zero_()

    def init_hidden_bmod(self, bs=None):
        weight_scale = 0.01
        if bs is None:
            bs = self.batch_size
        weight = next(self.parameters()).data
        return var(weight.new(self.n_layers, bs,
                              self.hidden_size)) \
            .uniform_(-weight_scale, weight_scale)

    def init_params(self, bs=None):
        if self.cell_type == "LSTM":
            self.hidden_c, = (self.init_hidden(bs), self.init_hidden(bs))
            self.hidden_q = (self.init_hidden(bs), self.init_hidden(bs))
            self.hidden_coatt = (self.init_hidden_coatt(
                bs), self.init_hidden_coatt(bs))
            self.hidden_bmod1 = (self.init_hidden_bmod(bs),
                                 self.init_hidden_bmod(bs))
        elif self.cell_type == "GRU":
            self.hidden_c, self.hidden_q = self.init_hidden(
                bs), self.init_hidden(bs)
            self.hidden_coatt = self.init_hidden_coatt(bs)
            self.hidden_bmod1 = self.init_hidden_bmod(bs)
        else:
            raise TypeError("Invalid Cell Type - use LSTM or GRU")

        if torch.cuda.is_available():
            self.beta = var(torch.ones((bs, self.c_size,))
                            * (1/self.c_size)).cuda()
        else:
            self.beta = var(torch.ones((bs, self.c_size,))*(1/self.c_size))

    def forward(self, inputs):
        if torch.cuda.is_available():
            if len(inputs) == 1:
                inputs = var(torch.cuda.LongTensor(inputs[0]))
            else:
                inputs = var(torch.cuda.LongTensor(inputs))
        else:
            if len(inputs) == 1:
                inputs = var(torch.LongTensor(inputs[0]))
            else:
                inputs = var(torch.LongTensor(inputs))

        if len(inputs.size()) < 2:
            inputs = inputs.unsqueeze(0)

        inputs_c = inputs[:, :self.c_size]
        inputs_q = inputs[:, self.c_size:self.c_size+self.q_size]
        embeds_c = self.encoder_c(inputs_c).permute(1, 0, 2)  # get glove repr
        embeds_q = self.encoder_q(inputs_q).permute(1, 0, 2)  # get glove repr

        c_op, self.hidden_c = self.gru_c(embeds_c, self.hidden_c)
        q_op, self.hidden_q = self.gru_q(embeds_q, self.hidden_q)

        q_op = F.softmax(self.lin_q(q_op), dim=0)

        # coattention network

        cq_op = torch.bmm(c_op.permute(1, 0, 2), q_op.permute(1, 2, 0))
        att_q = F.softmax(cq_op, dim=1)
        # transpose while keeping batch
        att_c = F.softmax(cq_op.permute(0, 2, 1), dim=1)
        contx_q = torch.bmm(att_q.permute(0, 2, 1), c_op.permute(1, 0, 2))
        q_contx_q = torch.cat((q_op, contx_q.permute(1, 0, 2)), 2)
        contx_c = torch.bmm(q_contx_q.permute(1, 2, 0), att_c)
        c_contx_c = torch.cat(
            (contx_c.permute(0, 2, 1), c_op.permute(1, 0, 2)), 2)
        coatt_op, self.hidden_coatt = self.gru_coatt(
            c_contx_c.permute(1, 0, 2), self.hidden_coatt)
        coatt_op = coatt_op.permute(1, 0, 2)  # revert to (N, L2, Hs# )
        # print("coatt op", coatt_op.shape)
        # answer pointer - boundary model

        # obtain start index
        f1 = F.dropout(self.ans_ptr_1(coatt_op), self.linear_dropout)
        H_beta = torch.bmm(self.beta.unsqueeze(1), coatt_op)
        if self.cell_type == "LSTM":
            self.hidden_bmod1, cell_bmod1 = self.hidden_bmod1  # hidden, cell

        # print("hbmod", self.hidden_bmod1.shape)
        aptr2 = self.ans_ptr_2(self.hidden_bmod1)
        # print("aptr2", aptr2.shape)
        W_h = F.dropout(aptr2.repeat(self.c_size, 1, 1), self.linear_dropout)
        # print(f1.shape, W_h.shape)
        f1 += W_h.permute(1, 0, 2)
        f1 = F.leaky_relu(f1)
        out_start = F.log_softmax(
            F.dropout(self.ans_ptr_3(f1), self.linear_dropout), 0).squeeze()

        # obtain end index
        f2 = F.dropout(self.ans_ptr_1(coatt_op), self.linear_dropout)
        if self.cell_type == "LSTM":
            _, (self.hidden_bmod2, cell_bmod2) = self.gru_bmod(
                H_beta.permute(1, 0, 2), (self.hidden_bmod1, cell_bmod1))
        else:
            _, self.hidden_bmod2 = self.gru_bmod(
                H_beta.permute(1, 0, 2), self.hidden_bmod1)
        W_h = F.dropout(self.ans_ptr_2(self.hidden_bmod2).repeat(
            self.c_size, 1, 1), self.linear_dropout)
        f2 += W_h.permute(1, 0, 2)
        f2 = F.leaky_relu(f2)
        out_end = F.log_softmax(
            F.dropout(self.ans_ptr_3(f2), self.linear_dropout), 0).squeeze()

        # put both together
        out = torch.cat((out_start, out_end), -1)
        if len(out.size()) < 2:
            out = out.unsqueeze(0)

        return out
