import torch
import torchtext.vocab as vocab


def setup_glove(name='6B', DIM=50):
    glove = vocab.GloVe(name='6B', dim=DIM)
    
    glove.stoi['<sos>'] = len(glove.stoi)+1
    glove.vectors = torch.cat((glove.vectors, torch.ones(1, DIM)*1))
    
    glove.stoi['<eos>'] = len(glove.stoi)+1
    glove.vectors = torch.cat((glove.vectors, torch.ones(1, DIM)*2))
    
    glove.stoi['<pad>'] = len(glove.stoi)+1
    glove.vectors = torch.cat((glove.vectors, torch.zeros(1, DIM)))
    
    glove.stoi['<unk>'] = len(glove.stoi)+1 # add token->index for unknown/oov
    glove.vectors = torch.cat((glove.vectors, torch.ones(1, DIM)*-1)) # add index->vec for unknown/oov
    
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
        
def token_idx_map(input_txt):
    input_seq = tokenize(input_txt)
    tok_idx_map = {"start": {}, "end": {}}
    curr_word = ""
    curr_word_idx = 0

    for i, c in enumerate(input_txt):
        if c != " ":
            curr_word += c
            try:
                input_tok = input_seq[curr_word_idx]
            except:
                input_tok = input_seq[curr_word_idx-1] #trailing spaces can cause this exception
            if curr_word == input_tok:
                s = i - len(curr_word) + 1
                e = i + 1 # since span is from [start, end)
                tok_idx_map["start"][s] = [curr_word_idx, curr_word]  # record what token starts here
                tok_idx_map["end"][e] = [curr_word_idx, curr_word] # record what token ends here
                curr_word = ""
                curr_word_idx += 1
    assert len(tok_idx_map["start"]) == len(tok_idx_map["end"])
    return tok_idx_map

def reverse_mapping(tok_idx_map):
    new_map = {}
    for k, v in tok_idx_map["start"].items():
        new_map[v[0]] = [v[1], k] # word, start
    for k, v in tok_idx_map["end"].items():
        assert k in new_map
        new_map[v[0]].append(k) # end
    return new_map

def get_answer_span(tok_idxs, input_txt):
    input_seq = tokenize(input_txt)
    s, e = tok_idxs
    return " ".join(input_seq[s:e+1])

def vectorize(input_seq, max_len):
    glove_vec = []
    glove_vec.append(400000) # <sos> token
    for w in input_seq:
        try:
            glove_vec.append(glove.stoi[clean(w)])
        except:
            glove_vec.append(400003) # <unk> token
    glove_vec = glove_vec[:max_len-1]
    glove_vec.append(400001) # <eos> token
    if len(glove_vec)<max_len:
        padding_zeros = [400002]*(max_len-len(glove_vec)) # <pad> token
        glove_vec = padding_zeros + glove_vec
    return glove_vec
    
def make_data(raw_X, raw_y):
    X = []
    y = []
    for i, ((qid, c, q, a), (s, e)) in enumerate(zip(raw_X, raw_y)):
        c_tokens = tokenize(c.lower())
        try:
            tok_idx_map = token_idx_map(c)
        except:
            print(c, q, a)
        #pprint(tok_idx_map)
        #pprint(reverse_mapping(tok_idx_map))
        try:
            start_tok_idx, start_w = tok_idx_map["start"][s]
        except:
            for i in range(s, e):
                if i in tok_idx_map["start"]: # get next tok
                    start_tok_idx, start_w = tok_idx_map["start"][i]
                    break
        try:
            end_tok_idx, _ = tok_idx_map["end"][e] #only idx, not token
        except:
            for i in range(e, s, -1):
                if i in tok_idx_map["end"]: # get prev tok
                    end_tok_idx, end_w = tok_idx_map["end"][i]
                    break
        context_rep = vectorize(c_tokens, 600)
        q_tokens = tokenize(q.lower())
        ques_rep = vectorize(q_tokens, 100)
        X.append(context_rep+ques_rep)
        y.append((start_tok_idx, end_tok_idx))
    return X, y