# Copy imports and preprocessing from generate.py
import matplotlib.pyplot as plt

import generate
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch.nn as nn


from lit_llama import model

import pandas as pd
import os

import gc # chasing mmeory leaks


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup

checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")


def load_LLaMA(checkpoint_path):
    with lazy_load(checkpoint_path) as checkpoint:
        name = llama_model_lookup(checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=None # We won't quantize the weights
        ):
            model = LLaMA.from_name(name)

        model.load_state_dict(checkpoint)
    return model

if __name__ == '__main__':

    print('Starting program')

    preprompt = '''Q: Roger has 5 tennis balls. He buys 2 more cans of
                tennis balls. Each can has 3 tennis balls. How many
                tennis balls does he have now?
                A: Roger started with 5 balls. 2 cans of 3 tennis balls
                each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
               '''

    query = '''Q: The cafeteria had 23 apples. If they used 20 to
                make lunch and bought 6 more, how many apples
                do they have?'''

    # Device init (copied from generate.py)
    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    

    

    LLaMA_config = model.LLaMAConfig.from_name('7B')
    print('Loading models...')
    # Load the LLaMa model and the IST generator (also a LLaMA model)
    LLamaModel = load_LLaMA(checkpoint_path)
    #LLamaModel = LLaMA(LLaMA_config)
    print('Finished loading the first model')
    print('Finished loading models')
    tokenizer = Tokenizer(tokenizer_path)
    alpaca_dataset = pd.read_json("alpaca_data_cleaned.json")
    
    IST_schemes = ['vanilla', 'last 4', '2nd to last', 'all layers']
    #IST_schemes = ['vanilla',]
    scheme_losses = {}
    

    for IST_scheme in IST_schemes:
        IST_generator = model.Block(LLaMA_config).to(fabric.device) 
        print(f'In scheme {IST_scheme}')
        print('Finished loading IST generator')
        '''
            Takes in a tensor of the shape (B, L, 4096) and outputs a tensor of the same shape
            B = batch size
            L = number of tokens in the sequence
            4096 = embedding dimension of each token
        '''

        # Define optimizer, loss function, tokenizer
        optimizer = torch.optim.Adam(IST_generator.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        scheme_losses[IST_scheme] = []

        for epoch in range(100):
            total_loss = 0
            print(f'In epoch {epoch}')

            for index, row in alpaca_dataset[:1].iterrows(): # for now, we'll only look at the first 200 items
                
                instruction = row['instruction']
                question = row['input']
                answer = row['output']
                
                # First, convert the preprompt into a single internal token
                
                encoded_instruction = tokenizer.encode(instruction, bos=True, eos=False, device=fabric.device).reshape((1,-1))
                encoded_query = tokenizer.encode(question, bos=True, eos=False, device=fabric.device)

                target = tokenizer.encode(answer, bos=True, eos=False, device=fabric.device)
                _, prelogits = LLamaModel(encoded_instruction) # prelogits is of the shape (B, L, 4096)
                
                del _, encoded_instruction

                if(IST_scheme == 'vanilla'):
                    ist_generator_out = IST_generator(prelogits) # (B, L, 4096)
                    internal_state_token = ist_generator_out[0, -1]
                elif(IST_scheme == 'last 4'):
                    internal_state_token = IST_generator.forward_last_4_hidden(prelogits) 
                elif(IST_scheme == '2nd to last'):
                    internal_state_token = IST_generator.forward_2nd_to_last_hidden(prelogits)
                elif(IST_scheme == 'all layers'):
                    internal_state_token = IST_generator.forward_sum_all_layers(prelogits)

                del prelogits

                # Next, send the internal token + query into LLaMA and save the output logits
                internal_state_token = internal_state_token.bfloat16() 
                '''
                    bug causes F.linear to crash if internal_state_token is not a bfloat16
                '''
                
                _, predicted_logits = generate.generate(model=LLamaModel,
                                                        idx=encoded_query,
                                                        max_new_tokens=target.size(0),
                                                        max_seq_length=2000, 
                                                        internal_state_tokens=internal_state_token,
                                                        top_k=2)

                print('tokenizer.decode(_): ', tokenizer.decode(_))
                print('Ground truth answer: ', answer)

                del _, internal_state_token

                '''
                encoded_combined = tokenizer.encode(preprompt + query, bos=True,eos=False, device=fabric.device)

                with torch.no_grad():
                    # Next, send the preprompt + query into LLaMA and save the output
                    idx, _ = generate.generate(model=model_and_IST_generator,
                                                idx=encoded_combined,
                                                max_new_tokens=100, 
                                                max_seq_length=2000, 
                                                internal_state_tokens=None)

                print('tokenizer.decode(idx): ', tokenizer.decode(idx))'''

                # Compare the two resulting outputs
            
                predicted_logits.requires_grad=True
                '''idx = idx[encoded_combined.size(0):]'''
                predicted_logits = predicted_logits.reshape(-1,32000)
                
                loss = loss_fn(predicted_logits.to(fabric.device), target.type(torch.LongTensor).to(fabric.device))
                print(f'loss: {loss}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.detach().cpu().numpy()
                
                del predicted_logits, target, loss

            scheme_losses[IST_scheme].append(total_loss)
            #if(epoch >= 1):
            #    os.remove(f'model_{IST_scheme}_{epoch-1}.pt')
            #torch.save(IST_generator.state_dict(), f'model_{IST_scheme}_{epoch}.pt')
            

        plt.plot(scheme_losses[IST_scheme])
        plt.xlabel('epoch')
        plt.ylabel('Cross entropy loss')
        plt.title(f'Loss while training on a single sample using scheme "{IST_scheme}"')

        plt.savefig(f'losses_{IST_scheme}.png')
        del IST_generator, optimizer, loss_fn

else:
    print(__name__)