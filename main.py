from experiment.experiment import Experiment
from tqdm import tqdm
from dataset.dataset_util import get_train_val_datasets
from configs.args import logger, args
import yaml
import os


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
                            dropout=None,
                            N=config['MODEL']['STACK_ENC_DEC'])

    if os.path.exists(args.checkpoint_file):
        loss, init = experiment.load_checkpoint(args.checkpoint_file)

    else:
        loss = 0
        init = 0

    a_batch = next(iter(train_data_loader))
    best_loss = 1e5

    for epoch in range(init, config['MODEL']['EPOCHS']):
        total_loss = 0
        batch_iterator = tqdm(a_batch, desc=f"Processing Epoch {epoch:02d}")
        # batch_iterator = tqdm(train_data_loader, desc=f"Processing Epoch {epoch:04d}")
        # for a_batch in batch_iterator:
        loss = experiment.train(a_batch)
        total_loss += loss
        batch_iterator.set_postfix({"loss": f"{loss:6.3f}"})

        # if loss<best_loss:
        #     best_loss = loss
        experiment.save_checkpoint(args.checkpoint_file, loss, epoch)


        # total_loss /= len(train_data_loader)


if __name__ == '__main__':

    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)

    main(configs)
