from model.transformer import Transformer
import torch
import torch.nn as nn
from configs.args import args, logger
from utils.model_utils import ModelUtil
import os


class Experiment(ModelUtil):
    def __init__(self, encoder_input_vocab_size: int, decoder_input_vocab_size: int, embedding_size, ff_hidden_layer, lr, weight_decay, l_s, source_tokenizer, target_tokenizer=None, head=6, dropout=None, N=6,):
        super(Experiment, self).__init__()

        self.model = Transformer(encoder_input_vocab_size=encoder_input_vocab_size, decoder_input_vocab_size=decoder_input_vocab_size, embedding_size=embedding_size,
                                 ff_hidden_layer=ff_hidden_layer, head=head, dropout=dropout, N=N)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay )
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=l_s, ignore_index=source_tokenizer.token_to_id('[PAD]'))

        if args.cpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda:0'

        for param in self.model.parameters():
            param.requires_grad = True


        # print(self.model)
        # exit()


    def train(self, a_batch):
        self.model.train()
        net = self.model.to(self.device)

        encoder_input_tensor = a_batch['encoder_input'].to(self.device)
        decoder_input_tensor = a_batch['decoder_input'].to(self.device)
        label_tensor = a_batch['label'].to(self.device)
        encoder_mask_tensor = a_batch['encoder_mask'].to(self.device)
        decoder_mask_tensor = a_batch['decoder_mask'].to(self.device)

        output = net(encoder_input_tensor, decoder_input_tensor, encoder_mask_tensor, decoder_mask_tensor)

        loss = self.criterion(output.view(-1, 22463), label_tensor.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation(self, validation_dataloader,):
        self.model.eval()
        with self.model.no_grad():
            pass
