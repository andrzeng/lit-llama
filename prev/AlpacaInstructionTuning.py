from pathlib import Path
import matplotlib.pyplot as plt
import lightning as L
import torch
import torch.nn as nn
from lit_llama import model
import random
from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup


fabric = L.Fabric(devices=1)
tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")
tokenizer = Tokenizer(tokenizer_path)

import json
with open('datasets/alpaca_data_cleaned.json') as f:
    alpaca_json = json.load(f)

# Create tokenized j
alpaca_train_tokens = []
alpaca_test_tokens = []

for item in alpaca_json[:5176]:
    alpaca_train_tokens.append(
        {
            'instruction': tokenizer.encode(item['instruction'], bos=False, eos=False, device=fabric.device),
            'input': tokenizer.encode(item['input'], bos=False, eos=False, device=fabric.device),
            'output':tokenizer.encode(item['output'], bos=False, eos=False, device=fabric.device)
        }
    )

for item in alpaca_json[5176:]:
    alpaca_test_tokens.append(
        {
            'instruction': tokenizer.encode(item['instruction'], bos=False, eos=False, device=fabric.device),
            'input': tokenizer.encode(item['input'], bos=False, eos=False, device=fabric.device),
            'output':tokenizer.encode(item['output'], bos=False, eos=False, device=fabric.device)
        }
    )


checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")

dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def load_LLaMA(checkpoint_path):
    with lazy_load(checkpoint_path) as checkpoint:
        name = llama_model_lookup(checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=None # We won't quantize the weights
        ):
            model = LLaMA.from_name(name)

        model.load_state_dict(checkpoint)
    return model


print('Loading models')

LLaMA_config = model.LLaMAConfig.from_name('7B')
print('Loading models...')
# Load the LLaMa model and the IST generator (also a LLaMA model)
LLamaModel = load_LLaMA(checkpoint_path).to(fabric.device)
#LLamaModel = LLaMA(LLaMA_config).to(fabric.device)
print('Finished loading the first model')
print('Finished loading models')
tokenizer = Tokenizer(tokenizer_path)

IST_schemes = ['vanilla', 'last 4', '2nd to last', 'all layers']
scheme_losses = {}

IST_generator = model.Block(LLaMA_config)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(IST_generator.parameters(), lr=1e-4)
IST_generator = IST_generator.to(fabric.device)

for param in LLamaModel.parameters():
    param.requires_grad=False

def get_single_example(dataset, index=None):
    if(index is None):
        index = random.sample(range(len(dataset)), k=1)[0]
    # IST
    IST = IST_generator(LLamaModel(dataset[index]['instruction'].unsqueeze(0).to(fabric.device))[1])[:,-1,:]

    # Question
    question = LLamaModel.transformer.wte(dataset[index]['input'].unsqueeze(0).to(fabric.device)).squeeze()

    # Answer fragment
    answer_len = dataset[index]['output'].size(0)
    trunc_len = random.randint(0,answer_len-1)
    #print(answer_len)
    #print(trunc_len)

    truncated_answer = dataset[index]['output'][:trunc_len]
    truncated_answer = LLamaModel.transformer.wte(truncated_answer)
    
    target_tokens = torch.cat([dataset[index]['input'], dataset[index]['output'][:trunc_len+1]])
    #print(tokenizer.decode(target_tokens))

    if(question.dim() == 1):
        question = question.unsqueeze(0)

    if(truncated_answer.dim() == 1):
        truncated_answer = truncated_answer.unsqueeze(0)

    llama_input = torch.cat([IST,question,truncated_answer])
    return llama_input.unsqueeze(0), target_tokens.type(torch.LongTensor).unsqueeze(0)
    
train_losses = []
test_losses = []

import wandb
learning_rate = 1e-4
batch_size=32
trainset_size=len(alpaca_train_tokens)
testset_size=len(alpaca_test_tokens)

config = {
    'batch_size': batch_size,
    'trainset_size': trainset_size,
    'testset_size':testset_size,
}

# init wandb
wandb.init(
    project='Alpaca instruction tuning',
    config=config,
)

optimizer = torch.optim.Adam(IST_generator.parameters(), lr=1e-5)


loss_fn = nn.CrossEntropyLoss()
for param in LLamaModel.parameters():
    param.requires_grad=False

batch_size=32

for epoch in range(10):
    indices = list(range(trainset_size))
    random.shuffle(indices)
    epoch_train_loss = 0
    
    while(len(indices) >= batch_size):
        batch_indices = indices[:batch_size]
        indices = indices[batch_size:]
        batch_loss = 0

        optimizer.zero_grad()
        for i in range(batch_size):
            input, target = get_single_example(alpaca_train_tokens, index=batch_indices[i])
            llama_output = LLamaModel.forward_embeddings(input.type(torch.bfloat16))[0]
            loss = loss_fn(llama_output.squeeze().to(fabric.device), target.squeeze().to(fabric.device))
            loss.backward()
            batch_loss += loss.item()
            del llama_output
            del loss
            del input, target

        batch_loss /= batch_size

        optimizer.step()
        train_losses.append(batch_loss)
        epoch_train_loss += batch_loss

        # validation:
        with torch.no_grad():
            batch_loss = 0
            for i in range(batch_size):
                input, target = get_single_example(alpaca_test_tokens)
                llama_output = LLamaModel.forward_embeddings(input.type(torch.bfloat16))[0]
                loss = loss_fn(llama_output.squeeze().to(fabric.device), target.squeeze().to(fabric.device))
                batch_loss += loss.item()
                del llama_output
                del loss
                
            batch_loss /= batch_size

            test_losses.append(batch_loss)
            
        print(f'epoch {epoch}, train loss = {train_losses[-1]}, validation loss={test_losses[-1]}')
        wandb.log({'batch train loss':train_losses[-1], 'batch validation loss':test_losses[-1], 'learning rate': optimizer.param_groups[0]['lr']})

    # perform validation on entire validation set
    with torch.no_grad():
        batch_loss = 0
        for i in range(testset_size):
            input, target = get_single_example(alpaca_test_tokens, i)
            llama_output = LLamaModel.forward_embeddings(input.type(torch.bfloat16))[0]
            loss = loss_fn(llama_output.squeeze().to(fabric.device), target.squeeze().to(fabric.device))
            del llama_output
            batch_loss += loss.item()
        batch_loss /= testset_size

    wandb.log({'epoch train loss':epoch_train_loss, 'epoch validation loss':batch_loss})


        
torch.save(IST_generator.state_dict(), "AlpacaTunedWeights.pt")