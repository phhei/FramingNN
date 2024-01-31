import random
from typing import Optional, Dict, List, Callable, Any

import numpy
import torch
import transformers
from torch.utils.data import Dataset
from pandas import DataFrame, read_csv
from transformers import PreTrainedTokenizer

from Utils import UserLabelCluster
from nltk import word_tokenize
from loguru import logger

# https://lightning.ai/docs/pytorch/stable/starter/installation.html
# https://lightning.ai/docs/torchmetrics/stable/pages/quickstart.html


def _preprocess_x(
        df: DataFrame,
        word_embeddings: Dict[str, numpy.ndarray],
        max_seq_len: Optional[int] = None,
        with_topic: bool = False
) -> List[numpy.ndarray]:
    """
    Preprocesses the textual input data (x) into a numpy list.
    :param df: The text instances
    :param word_embeddings: the word embeddings dictionary
    :param max_seq_len: maximum sequence length (if not given => no truncation)
    :param with_topic: should the topic be included in the input to the LM?
    :return: A numpy list (one list element for each instance, may be of different lengths)
    """
    x = [numpy.stack(
        (array := [word_embeddings[token] for token in word_tokenize(
            f"{'{}: '.format(row['topic']) if with_topic else ''}{row['premise']} => {row['conclusion']}"
        ) if token in word_embeddings])[:min(len(array), max_seq_len) if max_seq_len is not None else len(array)],
        axis=0
    ) for _, row in df.iterrows()]

    logger.trace("Preprocessed {} sequences (text into static word-embedding sequences)", len(x))

    return x


def process_x(
        df: DataFrame,
        word_embeddings: Dict[str, numpy.ndarray],
        max_seq_len: Optional[int] = None,
        with_topic: bool = False
) -> torch.Tensor:
    """
    Processes the textual input data (x) into a tensor (e.g., for CNNs/ linear layers).
    :param df: The text instances
    :param word_embeddings: the word embeddings dictionary
    :param max_seq_len: maximum sequence length (if not given => no truncation)
    :param with_topic: should the topic be included in the input to the LM?
    :return: A padded tensor of shape (batch_size, max_seq_len, embedding_dim)
    """
    x = _preprocess_x(df, word_embeddings, max_seq_len, with_topic)
    return torch.nn.utils.rnn.pad_sequence([torch.from_numpy(_x) for _x in x], batch_first=True, padding_value=-1)


def process_x_rnn(
        df: DataFrame,
        word_embeddings: Dict[str, numpy.ndarray],
        max_seq_len: Optional[int] = None,
        with_topic: bool = False
) -> torch.nn.utils.rnn.PackedSequence:
    """
    Processes the textual input data (x) into a tensor (e.g., for RNNs).

    :param df: The text instances
    :param word_embeddings: the word embeddings dictionary
    :param max_seq_len: maximum sequence length (if not given => no truncation)
    :param with_topic: should the topic be included in the input to the LM?
    :return: A packed tensor of shape (batch_size, max_seq_len, embedding_dim)
    """
    x = _preprocess_x(df, word_embeddings, max_seq_len, with_topic)
    lengths = [_x.shape[0] for _x in x]

    logger.debug("Received {} sequences with word amounts: {}--{}--{}",
                 len(x), min(lengths), round(sum(lengths) / len(lengths), 1), max(lengths))

    # https://stackoverflow.com/questions/73061601/how-to-mask-padded-0s-in-pytorch-for-rnn-model-training
    x_rnn = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(numpy.concatenate(([[_l]*_x.shape[-1]], _x), axis=0)) for _l, _x in zip(lengths, x)], batch_first=True, padding_value=0)

    return {"input": x_rnn, "lengths": lengths, "batch_first": True, "enforce_sorted": False}
    # return torch.nn.utils.rnn.pack_padded_sequence(x_rnn, lengths=lengths, batch_first=True, enforce_sorted=False)


def process_x_llm(
        df: DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: Optional[int] = None,
        with_topic: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Processes the textual input data (x) into a tensor (e.g., for a LLM (huggingdace)).
    :param df: The text instances
    :param tokenizer: the tokenizer of the LLM which should process the input later on
    :param max_seq_len: maximum sequence length (if not given => no truncation)
    :param with_topic: should the topic be included in the input to the LM?
    :return: A dict with tensor of shape (batch_size, max_seq_len)
    """
    texts_part1 = [f"{'{}: '.format(row['topic']) if with_topic else ''}{row['premise']}" for _, row in df.iterrows()]
    texts_part2 = [row['conclusion'] for _, row in df.iterrows()]

    logger.trace("Preprocessed {} sequences", len(df))

    logger.trace("{} will do the rest.", tokenizer.__class__.__name__)
    return tokenizer(
        text=texts_part1,
        text_pair=texts_part2,
        padding=True,
        truncation=transformers.tokenization_utils_base.TruncationStrategy.ONLY_FIRST,
        max_length=max_seq_len or 500,
        return_tensors="pt"
    )


def process_y_categorical(df: DataFrame, categories: Dict[str, int], frame_kind: str = "genericFrame") -> torch.Tensor:
    """
    Processes the textual target data (y) into a categorical tensor
    :param df: the text instances
    :param categories: the categories dictionary (frame set)
    :param frame_kind: "genericFrame" or "fuzzyFrame". ATTENTION: "fuzzyFrame" is not supported yet!
    :return: A tensor of shape (batch_size)
    """
    encodings_y = []
    lower_categories = {k.lower(): v for k, v in categories.items()}

    for i, row in df.iterrows():
        if row[frame_kind] not in categories:
            logger.warning("Category \"{}\" of instance \"{}\" not found in categories", row[frame_kind], i)
            logger.debug("Categories: {}", "|".join(categories.keys()))
            if row[frame_kind].lower() in lower_categories:
                logger.info("Category \"{}\" of instance \"{}\" found in categories (BUT case-insensitive)",
                            row["genericFrame"], i)
                encodings_y.append(lower_categories[row[frame_kind].lower()])
            else:
                choice = random.choice(list(categories.values()))
                logger.error("Category \"{}\" of instance \"{}\" not found in categories (case-insensitive),"
                             " we have to guess here! ({})",
                             row[frame_kind], i, choice)
                encodings_y.append(choice)
        else:
            logger.trace("Category \"{}\" of instance \"{}\" found in categories: {}",
                         row["genericFrame"], i, categories[row[frame_kind]])
            encodings_y.append(categories[row["genericFrame"]])

    return torch.tensor(encodings_y, dtype=torch.long)


def process_y_cluster(df: DataFrame, cluster: UserLabelCluster, frame_kind: str = "genericFrame") -> torch.Tensor:
    """
    Processes the textual target data (y) into a categorical tensor following a cluster
    :param df: the text instances
    :param cluster: the cluster object
    :param frame_kind: "genericFrame" or "fuzzyFrame". ATTENTION: "fuzzyFrame" is not supported yet!
    :return: A tensor of shape (batch_size)
    """

    encodings_y = [numpy.argmax(cluster.get_y(y)) for y in df[frame_kind]]

    return torch.tensor(encodings_y, dtype=torch.long)


class DfDataset(Dataset):
    def __init__(
            self,
            df: DataFrame,
            x_fc: Callable,
            x_params: Dict[str, Any],
            y_fc: Callable,
            y_params: Dict[str, Any],
            batch_size: int = 128
    ):
        super().__init__()
        self.df = df

        logger.debug("Dataset initialized with {} instances", len(self.df))
        self.x_fc = x_fc
        self.x_params = x_params
        self.y_fc = y_fc
        self.y_params = y_params
        self.batch_size = batch_size

        self.x = dict()
        self.y = dict()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx not in self.x:
            logger.debug("Preprocessing batch {}-{}...", idx, min(idx+self.batch_size, len(self.df)))
            self.x = {idx+i: v for i, v in enumerate(
                self.x_fc(self.df[idx:min(idx+self.batch_size, len(self.df))], **self.x_params)
            )}
            self.y = {idx + i: v for i, v in enumerate(
                self.y_fc(self.df[idx:min(idx + self.batch_size, len(self.df))], **self.y_params)
            )}

        return self.x[idx], self.y[idx]
