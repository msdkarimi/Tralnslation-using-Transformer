from experiment.experiment import Experiment
from tqdm import tqdm
from dataset.dataset_util import get_train_val_datasets



if __name__ == '__main__':

    train_data_loader, validation_data_loader, \
        source_vocab_size, target_vocab_size, \
        source_tokenizer, target_tokenizer = get_train_val_datasets()
    emb = 64
    experiment = Experiment(source_vocab_size, target_vocab_size, embedding_size=emb, ff_hidden_layer=emb*2,
                            source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer,
                            head=8, dropout=None)

    # a_batch = next(iter(train_data_loader))

    total_loss = 0
    init = 0
    epochs = 100
    for epoch in range(init, epochs):
        # batch_iterator = tqdm(a_batch, desc=f"Processing Epoch {epoch:02d}")
        batch_iterator = tqdm(train_data_loader, desc=f"Processing Epoch {epoch:04d}")
        for a_batch in batch_iterator:
            loss = experiment.train(a_batch)
            total_loss += loss
            batch_iterator.set_postfix({"loss": f"{loss:6.3f}"})

        total_loss /= len(train_data_loader)


    # print(output.shape)
    # prediction = torch.argmax(output, dim=-1)
    # tokens = prediction.squeeze(0).tolist()
    # words = bpemb_it.decode_ids(tokens)
    # print(words)











