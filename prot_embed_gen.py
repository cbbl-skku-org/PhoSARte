from transformers import (
    BertModel,
    BertTokenizer,
    T5Tokenizer,
    T5EncoderModel,
    AlbertTokenizer,
    AlbertModel,
    XLNetTokenizer,
    XLNetModel
)
import torch
import pandas as pd
import re

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

input_file = "./examples/samples.txt"

headers = []
sequences = []
labels = []
with open(input_file, "r") as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        sequence = lines[i+1].strip()
        headers.append(header[1:])
        sequences.append(sequence)
        if "Positive" in header:
            labels.append(1)
        else:
            labels.append(0)

data = pd.DataFrame({"IDs": headers, "Sequences": sequences, "Labels": labels})
print(data.head())

# Pretrained ProtTrans models: ['prot_bert_bfd', 'prot_t5_xl_bfd', 'prot_xlnet', 'prot_albert', 'prot_t5_xl_uniref50', 'prot_t5_xxl_uniref50', 'prot_bert', 'prot_t5_xxl_bfd']
# Replace with prot_t5_xxl_bfd with the others
# If T5 models:
#   tokenizer = T5Tokenizer
#   model = T5EncoderModel
# Elif BERT models:
#   tokenizer = BertTokenizer
#   model = BertModel
# Elif XLNet model:
#   tokenizer = XLNetTokenizer
#   model = XLNetModel
# Else:
#   tokenizer = AlbertTokenizer
#   model = AlbertModel
# Examples:
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
model = model.to(device)

torch_saved_file = "./embeddings/samples_prot_bert.pt"

def encode_sequence(sequence):
    sequence = " ".join(list(sequence)).upper()
    sequence = re.sub(r"[UZOB]", "X", sequence)
    
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=33
    )
    
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.squeeze(0) 
    
    embeddings = embeddings[:33]
    
    return embeddings.cpu()

embeddings_data = []

for i in range(len(data)):
    sequence = data.iloc[i]["Sequences"]
    embedding = encode_sequence(sequence)
    embeddings_data.append({
        'idx': data.iloc[i]['IDs'],
        'sequence': sequence,
        'embedding': embedding,
        'label': data.iloc[i]['Labels']
    })

torch.save(embeddings_data, torch_saved_file)

loaded_embeddings = torch.load(torch_saved_file)
print(loaded_embeddings[0])
print(loaded_embeddings[0]['embedding'].shape)
print(loaded_embeddings[0]['idx'])
print(loaded_embeddings[0]['sequence'])
print(loaded_embeddings[0]['embedding'])
print(loaded_embeddings[0]['label'])
