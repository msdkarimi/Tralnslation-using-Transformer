import torch

from experiment import *


class Experiment(ModelUtil):
    def __init__(self, encoder_input_vocab_size: int, decoder_input_vocab_size: int, embedding_size, max_seq_len: int, ff_hidden_layer, lr, weight_decay, l_s, source_tokenizer, target_tokenizer=None, head=6, dropout=None, N=6,):

        self.model = Transformer(encoder_input_vocab_size=encoder_input_vocab_size,
                                 decoder_input_vocab_size=decoder_input_vocab_size,
                                 embedding_size=embedding_size,
                                 max_seq_len=max_seq_len,
                                 ff_hidden_layer=ff_hidden_layer,
                                 head=head,
                                 dropout=dropout,
                                 N=N)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # TODO get config from config.yml
        self.scheduler = StepLR(self.optimizer, step_size=7, gamma=0.8)

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=l_s, ignore_index=source_tokenizer.token_to_id('[PAD]'))

        self.src_tknzr = source_tokenizer
        self.tgt_tknzr = target_tokenizer

        self.max_seq_len = max_seq_len
        self.decoder_input_vocab_size = decoder_input_vocab_size

        # evaluation metrics
        self.bleu = BLEUScore()
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()

        if args.cpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda:0'

        for param in self.model.parameters():
            param.requires_grad = True

    def train(self, a_batch):
        self.model.train()
        net = self.model.to(self.device)

        encoder_input_tensor = a_batch['encoder_input'].to(self.device)
        decoder_input_tensor = a_batch['decoder_input'].to(self.device)
        label_tensor = a_batch['label'].to(self.device)
        encoder_mask_tensor = a_batch['encoder_mask'].to(self.device)
        decoder_mask_tensor = a_batch['decoder_mask'].to(self.device)

        output = net(encoder_input_tensor, decoder_input_tensor, encoder_mask_tensor, decoder_mask_tensor)

        loss = self.criterion(output.view(-1, self.decoder_input_vocab_size), label_tensor.view(-1))
        # loss = self.criterion(output.view(-1, 22463), label_tensor.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation(self, a_batch):
        self.model.eval()
        with torch.no_grad():
            net = self.model.to(self.device)
            encoder_input_tensor = a_batch['encoder_input'].to(self.device)
            decoder_input_tensor = a_batch['decoder_input'].to(self.device)
            label_tensor = a_batch['label'].to(self.device)
            encoder_mask_tensor = a_batch['encoder_mask'].to(self.device)
            decoder_mask_tensor = a_batch['decoder_mask'].to(self.device)

            output = net(encoder_input_tensor, decoder_input_tensor, encoder_mask_tensor, decoder_mask_tensor)
            tmp = output.view(-1, self.decoder_input_vocab_size)
            pred = torch.argmax(tmp, dim=-1)
            text_lenght = torch.where(label_tensor.view(-1) == 2)[0].squeeze(-1)
            predictions = pred[:text_lenght.item()+1]
            predicted_text = self.tgt_tknzr.decode(predictions.detach().cpu().numpy())

            loss = self.criterion(output.view(-1, self.decoder_input_vocab_size), label_tensor.view(-1))

            return loss.item(), predicted_text

    def autoregressive_validation(self, a_batch,):
        self.model.eval()
        with torch.no_grad():

            net = self.model.to(self.device)
            predicts = list()
            expected = list()

            encoder_input = a_batch['encoder_input'].to(self.device)

            encoder_mask = a_batch['encoder_mask'].to(self.device)

            # decoder_mask_tensor = a_batch['decoder_mask'].to(self.device)

            encoded_input = net.encoding(encoder_input, encoder_mask) # shape 200*256

            def generate_text_based_on_trained_model():
                sos_token = self.tgt_tknzr.token_to_id('[SOS]')
                eos_token = self.tgt_tknzr.token_to_id('[EOS]')

                # encoded_input = net.encoding(encoder_input, encoder_mask)

                decoder_input = torch.empty(1, 1).fill_(sos_token).type_as(encoder_input).to(self.device)

                while True:
                    if decoder_input.shape[1] == self.max_seq_len:
                        break

                    decoder_mask = BilingualDataset.mask(decoder_input.shape[1]).type_as(encoder_mask).to(self.device)

                    generated_by_model_decoded = net.decoding(decoder_input, encoded_input, decoder_mask, encoder_mask)

                    prob_prediction = net.projection(generated_by_model_decoded[:, -1])

                    next_token = torch.argmax(prob_prediction, dim=-1)
                    # _, next_token = torch.max(prob_prediction, dim=1)

                    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_token.item()).to(self.device)], dim=1)

                    if next_token == eos_token:
                        break

                decoder_generated_tokens = decoder_input.squeeze(0)

                # print(decoder_generated_tokens)
                generated_text = self.tgt_tknzr.decode(decoder_generated_tokens.detach().cpu().numpy())
                return generated_text
            # source_text = a_batch["src_txt"][0]

            target_text = a_batch["tgt_txt"][0]
            predicted_text = generate_text_based_on_trained_model()

            expected.append(target_text)
            predicts.append(predicted_text)

            return f'in_eng: {a_batch["src_txt"][0]}\n'\
                f'expected : {expected}\n'\
                f'predicted: {predicts}'
                # f'tensor: {a_batch["decoder_input"]}'


            # return {
            #     'CER': self.cer(predicts, expected),
            #     'WER': self.wer(predicts, expected),
            #     'BLEU': self.bleu(predicts, expected),
            #     'in_eng': a_batch["src_txt"][0],
            #     'expected': expected,
            #     'predicted': predicts
            # }







