import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
# helps to create vocab based on list of sentences
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import random_split, DataLoader, Dataset
from dataset.bilingual_dataset import BilingualDataset, mask
from pathlib import Path


def get_build_tokenizer(ds, lang):

    tokenizer_path = Path('tokenizer/for_lang_{}'.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentence_form_dataset(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_all_sentence_form_dataset(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_train_val_datasets(lang_source, lang_target, batch_size, max_len, workers):
    torch.manual_seed(1371)
    ds_raw = load_dataset('opus_books', f'{lang_source}-{lang_target}', split='train')
    tokenizer_source = get_build_tokenizer(ds_raw, lang_source)
    tokenizer_target = get_build_tokenizer(ds_raw, lang_target)

    source_vocab_size = tokenizer_source.get_vocab_size()
    target_vocab_size = tokenizer_target.get_vocab_size()

    train_split_size = int(0.9 * len(ds_raw))
    val_split_size = len(ds_raw) - train_split_size


    train_ds, val_ds = random_split(ds_raw, [train_split_size, val_split_size])

    train_dataset = BilingualDataset(train_ds, tokenizer_source, tokenizer_target, 'en', 'it', seq_len=max_len)
    val_dataset = BilingualDataset(val_ds, tokenizer_source, tokenizer_target, 'en', 'it', seq_len=max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers)

    return train_dataloader, val_dataloader, source_vocab_size, target_vocab_size, tokenizer_source, tokenizer_target
