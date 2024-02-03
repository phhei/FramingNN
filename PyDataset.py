import random
from typing import Optional, Dict, List, Callable, Any, Union

import numpy
import torch
import transformers
from lightning.pytorch.utilities import CombinedLoader
from loguru import logger
from nltk import word_tokenize
from pandas import DataFrame
from torch.utils.data import Dataset, TensorDataset, StackDataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding

from Utils import UserLabelCluster


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
    x = []
    for arg_id, row in tqdm(df.iterrows(),
                            total=len(df), desc="Preprocessing text (W2V)", unit="instance", colour="yellow"):
        try:
            x.append(numpy.stack(
                (array := [word_embeddings[token] for token in word_tokenize(
                    f"{'{}: '.format(row['topic']) if with_topic else ''}{row['premise']} => {row['conclusion']}"
                ) if token in word_embeddings])[:min(len(array), max_seq_len) if max_seq_len is not None else len(array)],
                axis=0
            )
            )
        except ValueError:
            logger.opt(exception=True).error(
                "Error while processing instance \"{}\" - text cannot be encoded. Dummy it", arg_id
            )
            x.append(numpy.ones((1, len(next(iter(word_embeddings.values()))))))

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
) -> Dict[str, Any]:
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
    x_rnn = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(_x) for _x in x], batch_first=True, padding_value=0)

    return {"input": x_rnn, "lengths": torch.tensor(lengths, dtype=torch.int64, device="cpu")}
    # return torch.nn.utils.rnn.pack_padded_sequence(x_rnn, lengths=lengths, batch_first=True, enforce_sorted=False)


def process_x_llm(
        df: DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: Optional[int] = None,
        with_topic: bool = False
) -> BatchEncoding:
    """
    Processes the textual input data (x) into a tensor (e.g., for a LLM (huggingface)).
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


def process(df: DataFrame,
            x_fc: Callable,
            x_params: Dict[str, Any],
            y_fc: Callable,
            y_params: Dict[str, Any]) -> Dataset:
    """
    Wraps a dataframe into a dataset
    :param df: the dataframe
    :param x_fc: how to convert the input data
    :param x_params: parameters for x_fc
    :param y_fc: how to convert the target data
    :param y_params: parameters for y_fc
    :return: dataset (input for PyDataset.finish_datasets)
    """
    data = x_fc(df, **x_params)
    if not isinstance(data, Dict) and not isinstance(data, transformers.BatchEncoding):
        data = {"x": data}

    data["y"] = y_fc(df, **y_params)

    return StackDataset(**data)


def finish_datasets(datasets: Union[Dataset, List[Dataset]], batch_size: int, shuffle: bool = False) -> Union[DataLoader, CombinedLoader]:
    """
    Finishes the datasets by creating a dataloader for each dataset and combining them into one dataloader.
    (Can be used to finish a single dataset for lightning-training as well)
    :param datasets: one or more datasets (e.g., output of process())
    :param batch_size: the batch size
    :param shuffle: shuffles the data?
    :return: An instance for PyModels.setup_train
    """
    if not isinstance(datasets, list):
        datasets = [datasets]

    logger.info("Creating DataLoader for {} datasets ({} instances)", len(datasets), sum(map(len, datasets)))
    dataloader = [DataLoader(dataset=d, batch_size=batch_size, shuffle=shuffle) for d in datasets]

    if len(dataloader) == 1:
        return dataloader[0]

    logger.debug("Combining {} dataloaders, then let's train", len(dataloader))
    return CombinedLoader(dataloader, mode="max_size_cycle")
