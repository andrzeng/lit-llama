import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class AlpacaDataset(Dataset):
    def __init__(self, filename, tokenizer, num_samples=200, num_tokens=100, pad_value=1):
        alpaca_dataset = pd.read_json(filename)

        self.instructions = []
        self.questions = []
        self.answers = []
        for index, row in alpaca_dataset[:num_samples].iterrows(): # for now, we'll only look at the first 200 items
            instruction = tokenizer.encode(row['instruction'])[:num_tokens]
            question = tokenizer.encode(row['input'])[:num_tokens]
            answer = tokenizer.encode(row['output'])[:num_tokens]

            if(instruction.size(0) < num_tokens):
                instruction = F.pad(instruction, (0, num_tokens-instruction.size(0)), value=pad_value)

            if(question.size(0) < num_tokens):
                question = F.pad(question, (0, num_tokens-question.size(0)), value=pad_value)

            if(answer.size(0) < num_tokens):
                answer = F.pad(answer, (0, num_tokens-answer.size(0)), value=pad_value)


            self.instructions.append(instruction)
            self.questions.append(question)
            self.answers.append(answer)

    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        return self.instructions[idx], self.questions[idx], self.answers[idx]
    

class SQuAD_Dataset(Dataset):
    def __init__(self, filename, tokenizer, num_samples=200, num_tokens=100):
        squad_dataset = pd.read_json(filename)
        self.contexts = []
        self.questions = []
        self.answers = []

        for index, row in squad_dataset.iterrows():
            for context_qas in row['data']['paragraphs']:
                for qa in context_qas['qas']:
                    if(not qa['is_impossible']):
                        context = tokenizer.encode(context_qas['context'])
                        question = tokenizer.encode(qa['question'])
                        answer = tokenizer.encode(qa['answers'][0]['text'])
                        
                        if(context.size(0) < num_tokens):
                            context = F.pad(context, (0, num_tokens-context.size(0)))

                        if(question.size(0) < num_tokens):
                            question = F.pad(question, (0, num_tokens-question.size(0)))

                        if(answer.size(0) < num_tokens):
                            answer = F.pad(answer, (0, num_tokens-answer.size(0)))

                        self.contexts.append(context)
                        self.questions.append(question)
                        self.answers.append(answer)

    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        return self.contexts[idx], self.questions[idx], self.answers[idx]