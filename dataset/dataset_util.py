import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
# helps to create vocab based on list of sentences
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import random_split, DataLoader
from dataset.bilingual_dataset import BilingualDataset
from pathlib import Path


def get_build_tokenizer(ds, lang):

    tokenizer_path = Path('dictionaries/for_lang_{}'.format(lang))
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
    # torch.manual_seed(1371)
    # torch.manual_seed(145678371)
    # torch.manual_seed(13711212)  # 1.7528884410858154 20  1.7019003629684448 26 1.5525379180908203 32 1.6161561012268066  36
 # 36
 #    torch.manual_seed(13511271)  # 6.2185869216918945  5.844141006469727 5.864141006469727 1.7522130012512207

    ds_raw = load_dataset('opus_books', f'{lang_source}-{lang_target}', split='train')
    tokenizer_source = get_build_tokenizer(ds_raw, lang_source)
    tokenizer_target = get_build_tokenizer(ds_raw, lang_target)

    # max_len_src = 0
    # max_len_tgt = 0
    #
    # for item in ds_raw:
    #     src_ids = tokenizer_source.encode(item['translation']['en']).ids
    #     tgt_ids = tokenizer_target.encode(item['translation']['it']).ids
    #     max_len_src = max(max_len_src, len(src_ids))
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print(f'Max length of source sentence: {max_len_src}')
    # print(f'Max length of target sentence: {max_len_tgt}')

    source_vocab_size = tokenizer_source.get_vocab_size()
    target_vocab_size = tokenizer_target.get_vocab_size()

    train_split_size = int(0.9 * len(ds_raw))
    val_split_size = len(ds_raw) - train_split_size

    train_ds, val_ds = random_split(ds_raw, [train_split_size, val_split_size])

    train_dataset = BilingualDataset(train_ds, tokenizer_source, tokenizer_target, 'en', 'it', seq_len=max_len)
    val_dataset = BilingualDataset(val_ds, tokenizer_source, tokenizer_target, 'en', 'it', seq_len=max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_dataloader, val_dataloader, source_vocab_size, target_vocab_size, tokenizer_source, tokenizer_target

def interactive_data(encoder_input_text: str , lang_source, lang_target, max_len):

    data = {'translation':
                {'en': encoder_input_text,
                 'it': 'nulla'
                 }
            }

    tokenizer_source = Tokenizer.from_file(str(Path('dictionaries/for_lang_{}'.format(lang_source))))
    tokenizer_target = Tokenizer.from_file(str(Path('dictionaries/for_lang_{}'.format(lang_target))))

    interactive_dataset = BilingualDataset(data, tokenizer_source, tokenizer_target, lang_source, lang_target, seq_len=max_len)
    train_dataloader = DataLoader(interactive_dataset, batch_size=1, shuffle=False, num_workers=1)
    return train_dataloader