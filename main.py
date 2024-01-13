from experiment.experiment import Experiment
from tqdm import tqdm
from dataset.dataset_util import get_train_val_datasets
from configs.args import logger, args
import yaml
import torch
from torch import masked_fill
import numpy as np
import os


def test():
    pad = 0

    src = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 0, 0]])
    src_mask = (src == pad).type(torch.int16).unsqueeze(-2).unsqueeze(-2)
    print(src_mask)
    print(src_mask.shape)
    print(src_mask)
    seq_len = src.shape[-1]
    bs = src.shape[0]

    # Multiheads
    heads = 1
    attn_shape = (bs, heads, seq_len, seq_len)

    attn_scores = torch.rand(attn_shape)
    print(attn_scores.shape)
    src_mask_bool = (src_mask == 1)
    print(src_mask_bool.shape)
    attn_scores_masked = attn_scores.masked_fill(src_mask == 1, value=-1e9)
    print(attn_scores_masked)


def main(config):

    train_data_loader, validation_data_loader, \
        source_vocab_size, target_vocab_size, \
        source_tokenizer, target_tokenizer = get_train_val_datasets(lang_source=config['TASK']['SOURCE'],
                                                                    lang_target=config['TASK']['TARGET'],
                                                                    batch_size=config['MODEL']['BATCH_SIZE'],
                                                                    max_len=config['MODEL']['MAX_SEQ_LEN'],
                                                                    workers=config['MODEL']['WORKERS'])

    experiment = Experiment(source_vocab_size,
                            target_vocab_size,
                            embedding_size=config['MODEL']['EMB_DIM'],
                            max_seq_len=config['MODEL']['MAX_SEQ_LEN'],
                            ff_hidden_layer=config['MODEL']['FF_HIDDEN_LAYER'],
                            lr=config['MODEL']['LR'],
                            weight_decay=float(config['MODEL']['WEIGHT_DECAY']),
                            l_s=config['MODEL']['LABEL_SMOOTHING'],
                            source_tokenizer=source_tokenizer,
                            target_tokenizer=target_tokenizer,
                            head=config['MODEL']['HEADS'],
                            N=config['MODEL']['STACK_ENC_DEC'])

    loss, init = experiment.load_checkpoint(args.checkpoint_file)

    # a_batch = next(iter(train_data_loader))
    best_loss = 1e5 if loss == 0 else loss
    print(f'init with loss: {best_loss}')
    print(f'started with lr: {experiment.get_lr()}')

    for epoch in range(init, config['MODEL']['EPOCHS']):
        total_loss = 0
        # batch_iterator = tqdm(a_batch, desc=f"Processing Epoch {epoch:02d}")
        batch_iterator = tqdm(train_data_loader, desc=f"epoch {epoch:03d}/{config['MODEL']['EPOCHS']:03d}")
        for a_batch in batch_iterator:
            loss = experiment.train(a_batch)
            total_loss += loss
            batch_iterator.set_postfix({"loss": f"{loss:6.3f}"})

        total_loss /= len(train_data_loader)

        if total_loss < best_loss:
            best_loss = total_loss
            experiment.save_checkpoint(args.checkpoint_file, best_loss, epoch, type='best')

        file_name = f'checkpoint_epoch_{epoch}.pth'
        experiment.save_checkpoint(os.path.join(os.getcwd(), 'outputDir', 'weights', file_name), total_loss, epoch)

        experiment.scheduler.step() 
        print(total_loss)
        print(f'next iteration lr will be : {experiment.get_lr()}')
    logger.info('}')




    batch_iterator = tqdm(validation_data_loader)
    for a_batch in batch_iterator:
        out = experiment.autoregressive_validation(a_batch)
        print(out)
        print("------------------------")


if __name__ == '__main__':

    # test()

    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)

    main(configs)
