from transformer.transformer_model import Transformer
import torch
import torch.nn as nn
from configs.args import args, logger
import os


class Experiment:
    def __init__(self, encoder_input_vocab_size: int, decoder_input_vocab_size: int, embedding_size, ff_hidden_layer, lr, l_s, source_tokenizer, target_tokenizer=None, head=6, dropout=None, N=6,):
        self.model = Transformer(encoder_input_vocab_size=encoder_input_vocab_size, decoder_input_vocab_size=decoder_input_vocab_size, embedding_size=embedding_size,
                                 ff_hidden_layer=ff_hidden_layer, head=head, dropout=dropout, N=N)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=l_s, ignore_index=source_tokenizer.token_to_id('[PAD]'))

        if args.cpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda:0'

        for param in self.model.parameters():
            param.requires_grad = True



        # # TODO just for initialization  not after loading
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def save_checkpoint(self, checkpoint_path, loss, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'epoch': epoch
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            logger.info('-----------------Loading checkpoint!')
            checkpoint = torch.load(checkpoint_path)

            # for key, value in checkpoint['model'].items():
            #     print(key)
            # exit(0)
            #
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            return checkpoint['loss'], checkpoint['epoch']

        else:
            raise FileExistsError(f'file {checkpoint_path} does not exists!')

    def validation(self, validation_dataloader,):
        self.model.eval()
        with self.model.no_grad():
            pass
