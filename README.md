# Tralnslation-using-Transfors
Educational repository, implementing encoder/decoder transformer model to perform translation
 # Attention Is All You Need
- [Linke to paper ](https://arxiv.org/pdf/1706.03762.pdf)

# Model
- Model itself is composed of [an encoder](model/encoder.py) and [a decoder](model/decoder.py). Based on the paper, each-one is consists of a few modules which mainly are [multi-head attention](utils/blocks_util.py), [residual connections plus layer normalisation](utils/blocks_util.py) and [feedforward network](utils/blocks_util.py).
- [tokenizers](https://huggingface.co/docs/tokenizers/index) package is used to perform tokenization
# Dataset
- Thanks to [hugging face](https://huggingface.co) , datasets package is used to load 'opus_books' dataset

# Evaluation

