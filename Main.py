import pickle
from collections import defaultdict
from typing import Tuple, Optional, Union, List, Dict, Any, TypeVar

import click
import pandas
import transformers
import torch

from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset

import Frames
import PyModels
import Utils
from PyDataset import (process_x, process_x_rnn, process_x_llm, process_y_categorical, process_y_cluster,
                       process, finish_datasets)

T = TypeVar("T")
input_fct_mapper = {
    "w2v": process_x,
    "general_w2v": process_x,
    "rnn_w2v": process_x_rnn,
    "rnn": process_x_rnn,
    "llm": process_x_llm,
    "transformer": process_x_llm
}
target_fct_mapper = {
    "categorical": process_y_categorical,
    "categorical_most_frequent": process_y_categorical,
    "categorical_all_mf_wo_other": process_y_categorical,
    "categorical_all": process_y_categorical,
    "cluster": process_y_cluster,
    "cluster_3": process_y_cluster,
    "cluster_5": process_y_cluster,
    "cluster_10": process_y_cluster,
    "cluster_15": process_y_cluster,
    "cluster_25": process_y_cluster,
    "cluster_100": process_y_cluster,
}


def ensure_list(x: Union[T, List[T]]) -> List[T]:
    if isinstance(x, list) or isinstance(x, tuple):
        logger.trace("Found a list with {} elements ({} are None)", len(x), sum(map(lambda _x: int(_x is None), x)))
        return None if len(x) == 0 or all(map(lambda _x: _x is None, x)) else x
    else:
        logger.debug("Convert single value to list")
        return [x]


@click.command(add_help_option=True)
@click.option(
    "--runs", "-r",
    default=1,
    show_default=True,
    type=click.IntRange(1, 100, clamp=True),
    help="How many runs do you want to perform? (in each run, the model parameters are initialized randomly)"
)
@click.option(
    "--output_root_path", "-out",
    default=None,
    show_default=False,
    help="You can specify the output root path where all results/ weights will be saved. "
         "If not set, a sensible path will be generated",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path)
)
@click.option(
    "--train_data_path", "-train",
    multiple=True,
    help="You can specify the path to the training data here (should be preprocessed). "
         "Define multiple paths (by setting several -train <value>) in case of multi-task learning",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--train_data_frac", "-trainfrac",
    multiple=True,
    nargs=2,
    type=click.Tuple([float, float]),
    default=None,
    show_default=False,
    help="You can specify the fraction to use as the training data here (between 0 and 1 as tuple). "
         "Define multiple paths (by setting several -train <value> <value>) in case of multi-task learning",
)
@click.option(
    "--dev_data_path", "-dev", "-val",
    multiple=True,
    help="You can specify the path to the development data here (should be preprocessed). "
         "Define multiple paths (by setting several -dev <value> <value>) in case of multi-task learning",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--dev_data_frac", "-devfrac", "-valfrac",
    multiple=True,
    nargs=2,
    type=click.Tuple([float, float]),
    default=None,
    show_default=False,
    help="You can specify the fraction to use as the development data here (between 0 and 1 as tuple). "
         "Define multiple paths (by setting several -devfract <value> <value>) in case of multi-task learning",
)
@click.option(
    "--test_data_path", "-test",
    multiple=True,
    help="You can specify the path to the testing data here (should be preprocessed). "
         "Define multiple paths (by setting several -test <value>) in case of multi-task learning",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--test_data_frac", "-testfrac",
    multiple=True,
    nargs=2,
    type=click.Tuple([float, float]),
    default=None,
    show_default=False,
    help="You can specify the fraction to use as the testing data here (between 0 and 1 as tuple). "
         "Define multiple paths (by setting several -testfrac <value> <value>) in case of multi-task learning",
)
@click.option(
    "--fct_input_process", "-in",
    multiple=True,
    help="You can specify the method to preprocess the text here. "
         "Define the same amount of multiple methods (by setting several -in <value>) in case of multi-task learning",
    type=click.Choice(list(input_fct_mapper.keys()), case_sensitive=True)
)
@click.option(
    "--fct_output_process", "-out", "-target",
    multiple=True,
    help="You can specify the method to preprocess the target (class) here. "
         "Define the same amount of multiple methods (by setting several -out <value>) in case of multi-task learning",
    type=click.Choice(list(target_fct_mapper.keys()), case_sensitive=True)
)
@click.option(
    "--process_topics", "-topics",
    default=False,
    is_flag=True,
    show_default=True,
    help="Should the topic be included as well in the input to the model?"
)
@click.option(
    "--max_length", "-length", "-len", "-l",
    default=None,
    show_default=True,
    type=int,
    help="Do you want to limit the maximum sequence length? (e.g., for RNNs) If yes, set a token limit (int) here"
)
@click.option(
    "--shuffle_data", "-shuffle",
    default=False,
    type=int,
    help="Should the data be shuffled? If yes, set a random seed (int) here"
)
@click.option(
    "--model", "-m",
    default="rnn",
    show_default=True,
    help="You can specify the model (type) to use here",
    type=click.Choice(["rnn", "gru", "lstm", "transformer"], case_sensitive=False)
)
@click.option(
    "--model_params", "-mp",
    default=None,
    show_default=True,
    multiple=True,
    help="You can specify the model parameters (as a dictionary) here",
    type=click.Tuple([str, str])
)
@click.option(
    "--batch_size", "-bs",
    default=64,
    show_default=True,
    type=click.IntRange(1, 128, clamp=True),
    help="Batch size for training"
)
@click.option(
    "--learning_rate", "-lr",
    default=1e-3,
    show_default=True,
    type=click.FloatRange(1e-10, 1., clamp=True),
    help="Learning rate for training"
)
@click.option(
    "--hard_parameter_sharing", "-hps",
    default=False,
    is_flag=True,
    show_default=True,
    help="(only relevant for multi-task learning) Should the models share their parameters hardly? "
         "(the core components are the same)"
)
@click.option(
    "--soft_parameter_sharing", "-sps",
    default=None,
    show_default=True,
    type=click.FloatRange(1e-3, 2., clamp=False),
    help="(only relevant for multi-task learning) Should the models share their parameters softly? "
         "(the core components are the same type and architecture, but different weights)"
)
@click.option(
    "--early_stopping", "-es",
    default=False,
    is_flag=True,
    show_default=True,
    help="Should the training be stopped early if the validation loss does not decrease anymore?"
)
@logger.catch(level="CRITICAL", message="An error occurred during the main process")
def run(runs: int,
        output_root_path: Optional[Path],
        train_data_path: Union[Path, List[Path]],
        train_data_frac: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]],
        dev_data_path: Union[Path, List[Path]],
        dev_data_frac: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]],
        test_data_path: Optional[Union[Path, List[Path]]],
        test_data_frac: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]],
        fct_input_process: Union[str, List[str]],
        fct_output_process: Union[str, List[str]],
        process_topics: bool,
        max_length: Optional[int],
        shuffle_data: Union[bool, int],
        model: str,
        model_params: Optional[Union[Dict[str, Any], List[Tuple[str, str]]]],
        batch_size: int,
        learning_rate: float,
        hard_parameter_sharing: bool,
        soft_parameter_sharing: Optional[float],
        early_stopping: bool = True) -> None:
    logger.debug("Let's get started!")

    # Variable type conversion
    if not isinstance(model_params, Dict):
        model_params = ensure_list(model_params)
        if model_params is not None:
            model_params = dict(model_params)
            # TODO: type conversion
    train_data_path = ensure_list(train_data_path)
    train_data_frac = ensure_list(train_data_frac)
    dev_data_path = ensure_list(dev_data_path)
    dev_data_frac = ensure_list(dev_data_frac)
    test_data_path = ensure_list(test_data_path)
    test_data_frac = ensure_list(test_data_frac)
    fct_input_process = ensure_list(fct_input_process)
    fct_output_process = ensure_list(fct_output_process)

    # Load data
    data_df = dict()
    for key, data_path, data_frac in [("train", train_data_path, train_data_frac),
                                      ("dev", dev_data_path, dev_data_frac),
                                      ("test", test_data_path, test_data_frac)]:
        if data_path:
            if data_frac:
                assert len(data_frac) == len(data_path)
            else:
                data_frac = [(0., 1.)] * len(data_path)
            data_df[key] = [
                pandas.read_csv(
                    dp,
                    sep="|",
                    encoding="utf-8",
                    encoding_errors="replace",
                    index_col="argument_id"
                )
                for dp in data_path
            ]
            # DON'T punish OTHER
            # data_df[key] = [
            #     df[df.genericFrame != "__UNKNOWN__"] if dp.parent.parent.name == "MediaFramesCorpus" else df
            #     for df, dp in zip(data_df[key], data_path)
            # ]
            data_df[key] = [
                df.iloc[int(frac[0]*len(df)):int(frac[1]*len(df))]
                for df, frac in zip(data_df[key], data_frac)
            ]
            if bool(shuffle_data):
                logger.debug("Shuffle data (seed: {})", shuffle_data if isinstance(shuffle_data, int) else "not given")
                data_df[key] = [
                    df.sample(
                        frac=1, replace=False, random_state=shuffle_data if isinstance(shuffle_data, int) else None
                    ) for df in data_df[key]
                ]
            logger.debug("Loaded {} {} dataframes ({} instances in total)",
                         len(data_df[key]), key, sum(map(len, data_df[key])))
        else:
            logger.info("No {} data provided", key)

    if "test" in data_df:
        logger.debug("Ensure that there is no overlap between train and test data")
        for test_df, train_df in zip(data_df["test"], data_df["train"]):
            if set(train_df.index).isdisjoint(set(test_df.index)):
                logger.trace("No overlap found")
            else:
                logger.warning("Overlap found between test and train data: {} instances",
                               len(set(train_df.index).intersection(set(test_df.index))))
                train_df.drop(index=test_df.index, inplace=True, errors="ignore")
                logger.info("Train data has {} instances now", len(train_df))

    # Preprocess data
    logger.debug("Preprocess data")

    def process_data(f_fct_input_process, f_fct_output_process, f_current_run: int = 1) -> Tuple[Dict[str, Dataset], int]:
        f_processed_data = defaultdict(list)
        f_num_classes = None
        for i, (fct_in, fct_out) in enumerate(zip(f_fct_input_process, f_fct_output_process)):
            if fct_in not in input_fct_mapper or fct_out not in target_fct_mapper:
                raise ValueError(f"Unknown function {fct_in}/{fct_out}")
            fct_in_name = input_fct_mapper[fct_in]
            fct_in_params = {"max_seq_len": max_length, "with_topic": process_topics}
            if fct_in_name == process_x_rnn or fct_in_name == process_x:
                fct_in_params["word_embeddings"] = Utils.load_word_embeddings(
                    glove_file=Path("../../_wordEmbeddings/glove/glove.840B.300d.txt"),
                    embedding_size=300
                )
            elif fct_in_name == process_x_llm:
                fct_in_params["tokenizer"] = transformers.AutoTokenizer.from_pretrained(
                    model_params.get("pretrained_model_name_or_path", "roberta-base")
                    if model_params else "roberta-base"
                )
            logger.debug("Preprocess {}. input data with function \"{}\" and parameters: {}",
                         i, fct_in,
                         ", ".join(map(lambda kv: f"{kv[0]} ({kv[1] if isinstance(kv[1], bool) or isinstance(kv[1], str) else type(kv[1])})",
                                       fct_in_params.items())))

            fct_out_name = target_fct_mapper[fct_out]
            fct_out_params = {"frame_kind": "genericFrame"}
            if fct_out == "categorical" or fct_out == "categorical_all":
                fct_out_params["categories"] = \
                    {name: i for i, name in enumerate(Frames.media_frames_set.frame_names + ["__UNKNOWN__"])}
                f_num_classes = len(Frames.media_frames_set.frame_names)+1
            elif fct_out == "categorical_all_mf_wo_other":
                fct_out_params["categories"] = \
                    {name: i for i, name in enumerate(Frames.media_frames_set.frame_names)}
                f_num_classes = len(Frames.media_frames_set.frame_names)
            elif fct_out == "categorical_most_frequent":
                fct_out_params["categories"] = \
                    {name: i for i, name in enumerate(Frames.most_frequent_media_frames_set.frame_names)}
                f_num_classes = len(Frames.most_frequent_media_frames_set.frame_names)
            elif fct_out.startswith("cluster"):
                f_num_classes = int(fct_out.split("_")[1]) if "_" in fct_out else 15
                cluster_file = Path("clusters/9860x300d_semantic_{}c_{}.pkl".format(f_num_classes, f_current_run))
                logger.info("Load cluster file for {}: {}", fct_out, cluster_file.absolute())
                with cluster_file.open(mode="rb") as f:
                    fct_out_params["cluster"] = pickle.load(f)
            logger.debug("Preprocess {}. target data with function \"{}\" and parameters: {}",
                         i, fct_out, fct_out_params)

            for split, data in data_df.items():
                logger.trace("Process {}=>{}. data", split, i)
                f_processed_data[split].append(
                    process(df=data[i],
                            x_fc=fct_in_name, x_params=fct_in_params,
                            y_fc=fct_out_name, y_params=fct_out_params)
                )
        f_final_processed_data = dict()
        for split, datas in f_processed_data.items():
            logger.debug("Finish data-split {}", split)
            f_final_processed_data[split] = finish_datasets(datasets=datas, batch_size=batch_size, shuffle=False)
        logger.success("Data is ready! Proposed classes: {}", f_num_classes or "unknown")
        return f_final_processed_data, f_num_classes

    final_processed_data, num_classes = process_data(
        f_fct_input_process=fct_input_process,
        f_fct_output_process=fct_output_process
    )

    for current_run in range(1, runs+1):
        try:
            logger.info("Let's start Run {}/{}", current_run, runs)

            # Define model
            logger.debug("Define model ({} given)", model)
            if model.lower() in ["rnn", "gru", "lstm"]:
                model_class = torch.nn.RNN if model.lower() == "rnn" else (
                    torch.nn.GRU) if model.lower() == "gru" else torch.nn.LSTM
                _model_params = {
                    "input_size": 300,
                    "hidden_size": 128,
                    "dropout": 0,
                    "num_layers": 1,
                    "bias": True,
                    "batch_first": True,
                    "bidirectional": True
                }
                _model_params.update(model_params or dict())
            elif model == "transformer":
                model_class = transformers.AutoModel
                _model_params = {
                    "pretrained_model_name_or_path": "roberta-base",
                    "return_dict": True
                }
                _model_params.update(model_params or dict())
            else:
                raise ValueError(f"Unknown model {model}")

            core_models = [model_class(**_model_params)]*len(train_data_path) \
                if hard_parameter_sharing else \
                [model_class(**_model_params) for _ in range(len(train_data_path))]
            logger.debug("Creates {} core model(s): {}", len(core_models), core_models[0])

            # Ensure actual data:
            if any(map(lambda f_o: f_o.startswith("cluster"), fct_output_process)):
                logger.debug("Ensure actual data for cluster model")
                final_processed_data, num_classes = process_data(
                    f_fct_input_process=fct_input_process,
                    f_fct_output_process=fct_output_process,
                    f_current_run=current_run
                )
                logger.trace("Final processed data done for run {} ({} classes)", current_run, num_classes)

            # Prepare the model
            logger.debug("Prepare the model for training")
            final_models = [PyModels.ClassificationModule(
                core_model=core_models[i],
                num_classes=num_classes,
                learning_rate=learning_rate,
                task_name=f"IN{fct_in_name.lower()}MODEL{model.lower()}OUT{fct_out_name.lower()}_{i}_Run{current_run}"
            ) for i, (fct_in_name, fct_out_name, core_model) in enumerate(
                zip(fct_input_process, fct_output_process, core_models)
            )]
            logger.debug("Creates {} final model(s) ({} classes in total)",
                         len(final_models), sum(map(lambda m: m.num_classes, final_models)))
            if len(final_models) == 1:
                logger.trace("Just one model to train (multi-task: off)\n => Final model: {}", final_models[0])
                final_train_object = final_models[0]
            else:
                final_train_object = PyModels.MultiClassificationModule(
                    classification_modules=final_models,
                    soft_parameter_sharing=soft_parameter_sharing,
                    learning_rate=learning_rate,
                    task_weighting=[len(dl)/max(map(len, final_processed_data["train"].iterables))
                                    for dl in final_processed_data["train"].iterables]
                    if hasattr(final_processed_data["train"], "iterables") else None,
                )
                logger.info("Multi-task training ({} models)\n => Final model: {}", len(final_models), final_train_object)

            # Train and test the model
            if output_root_path is None:
                output_root_path = Path(".out").joinpath(
                    "{}ON{}".format("+".join(map(lambda p: p.stem.lower(), train_data_path)),
                                    "+".join(map(lambda p: p.stem.lower(), test_data_path)))
                ).joinpath(
                    "{}2{}{}".format(
                        "_".join(fct_input_process),
                        "_".join(fct_output_process),
                        "" if max_length is None else f"_max{max_length}tokens"
                    )
                ).joinpath(
                    model
                ).joinpath(
                    f"lr{learning_rate}_bs{batch_size}_"
                    f"sharing{'All' if hard_parameter_sharing else ('None' if soft_parameter_sharing is None else soft_parameter_sharing)}_"
                    f"{'earlystopped' if early_stopping else ''}"
                )
            output_root_path = output_root_path.joinpath(f"Run{current_run:>02}")
            output_root_path.mkdir(parents=True, exist_ok=True)
            logger.info("Finally, train the model (\"{}\")", output_root_path.name)
            logger.debug("Stores logging/ weights to: \"{}\"", output_root_path.absolute())
            PyModels.setup_train(
                module=final_train_object,
                training_data=final_processed_data["train"],
                validation_data=final_processed_data["dev"],
                test_data=final_processed_data.get("test"),
                root_path=output_root_path,
                monitoring_metric=f"val_f1Micro_{final_models[0].task_name}" if early_stopping else None
            )
            logger.success("DONE (Run {}/{})", current_run, runs)
        except Exception:
            logger.opt(exception=True).error("An error occurred during run {} (skip this run)", current_run)


if __name__ == "__main__":
    run()
