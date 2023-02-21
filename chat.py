import transformers
import torch
import pickle
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
from transformers import BertForSequenceClassification,BertTokenizer
class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,cache_dir="bert_cache_dir/")

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea
    def embed(self,text):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir="bert_cache_dir/")
        return tokenizer.encode(text)


def load_opening_w_LR():
    model = transformers.BertModel.from_pretrained('bert-base-uncased',cache_dir="bert_cache_dir/")
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased',cache_dir="bert_cache_dir/")
    loaded_ML_model = pickle.load(open("CO_LR.sav", 'rb'))
    return model, tokenizer, loaded_ML_model

def load_opening_w_bert(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location='cpu')
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    #return state_dict['valid_loss']
# Generate embeddings for the texts in the dataset
def generate_embeddings(texts,tokenizer,model):
    input_ids = tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt')['input_ids']
    with torch.no_grad():
        last_hidden_states = model(input_ids).last_hidden_state
    embeddings = last_hidden_states.mean(dim=1)
    return embeddings
def co(text, classifer):
    clifer = get_classifer(classifer,label = 'co')

    out = clifer.predict(text)

    return bool(out)



