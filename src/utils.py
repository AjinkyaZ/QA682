from pprint import pprint
import torch
import torchtext.vocab as vocab


def setup_glove(name='6B', DIM=50):
    glove = vocab.GloVe(name='6B', dim=DIM)

    glove.stoi['<sos>'] = len(glove.stoi)+1  # 400000
    glove.vectors = torch.cat((glove.vectors, torch.ones(1, DIM)*1))

    glove.stoi['<eos>'] = len(glove.stoi)+1  # 4000001
    glove.vectors = torch.cat((glove.vectors, torch.ones(1, DIM)*2))

    glove.stoi['<pad>'] = len(glove.stoi)+1  # 400002
    glove.vectors = torch.cat((glove.vectors, torch.zeros(1, DIM)))

    # 400003 - add token->index for unknown/oov
    glove.stoi['<unk>'] = len(glove.stoi)+1
    # add index->vec for unknown/oov
    glove.vectors = torch.cat((glove.vectors, torch.ones(1, DIM)*-1))

    return glove


def clean(token):
    cleaned_token = token.strip(".,?!-:;'()[]\"`")
    if cleaned_token[-2:] == "'s":
        cleaned_token = cleaned_token[:-2]
    if cleaned_token[-2:] == "'t":
        cleaned_token = cleaned_token[:-2]+'t'
    return cleaned_token


def tokenize(input_txt):
    return [w for w in input_txt.split(" ") if len(clean(w).strip())]


def get_answer_span(tok_idxs, padding, input_txt):
    input_seq = tokenize(input_txt)
    s, e = tok_idxs
    return " ".join(input_seq[s-padding:e-padding+1])


def vectorize(input_seq, max_len, glove):
    glove_vec = []
    glove_vec.append(400000)  # <sos> token
    for w in input_seq:
        try:
            glove_vec.append(glove.stoi[clean(w)])
        except:
            glove_vec.append(400003)  # <unk> token
    glove_vec = glove_vec[:max_len-1]
    glove_vec.append(400001)  # <eos> token
    if len(glove_vec) < max_len:
        padding_zeros = [400002]*(max_len-len(glove_vec))  # <pad> token
        glove_vec = padding_zeros + glove_vec
    return glove_vec


def make_data(raw_X, raw_y, max_length, glove):
    X = []
    y = []
    idxs = []
    pad_length = []
    skipped = 0
    for i, ((qid, c, q, a), (s, e)) in enumerate(zip(raw_X, raw_y)):
        start_tok_idx = len(c[:c.find(a)].split())
        end_tok_idx = start_tok_idx+(len(a.split()))-1
        c_tokens = tokenize(c.lower())
        context_rep = vectorize(c_tokens, 600, glove)
        q_tokens = tokenize(q.lower())
        ques_rep = vectorize(q_tokens, 100, glove)
        try:
            padlen = context_rep.count(400002)+1
            span = (start_tok_idx+padlen, end_tok_idx+padlen)
            if start_tok_idx+padlen >= 600 or end_tok_idx+padlen >= 600:
                print("Error: index overflow")
                #print("Problem idx:", i, ", qid:", qid)
                #print("span chars:", (s, e))
                #print("padlen", padlen)
                #print("span tokens (w/o) padlen:", (start_tok_idx, end_tok_idx))
                #print("span tokens (w/ padlen):", (start_tok_idx+padlen, end_tok_idx+padlen))
                #print("length of passage:", len(c_tokens))
                # print(c)
                # print(a)
                skipped += 1
                continue
            y.append(span)
            X.append(context_rep+ques_rep)
            pad_length.append(padlen)
            if span[0] >= 600 or span[1] >= 600 or span[0] < 0 or span[1] < 0:
                print(span)
            if start_tok_idx+padlen < 0 or end_tok_idx+padlen < 0:
                print("Error: index underflow")
                print(start_tok_idx+padlen, end_tok_idx+padlen)
                skipped += 1
                continue
            idxs.append(i)
        except:
            skipped += 1
            continue
        if len(X) == len(y) and len(X) < max_length:
            continue
        else:
            break
    if skipped:
        print("Skipped:", skipped)
    return idxs, pad_length, X, y