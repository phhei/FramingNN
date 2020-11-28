"""
Utils for the approaches
"""
import math
from typing import Union, Optional, Tuple, List, Dict

# We executed the code on strong CPU clusters without an GPU (ssh compute). Because of this extraordinary
# executing environment, we introduce this flag. To reproduce the results in the paper, enable this flag.
execute_on_ssh_compute = False

import csv
import os
import pathlib
import pickle
import random
import datetime
from functools import reduce
import word_mover_distance.model as word_mover_distance

import loguru
import tensorflow
import numpy
from tensorflow import keras

import nltk

nltk.download("punkt")
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
from nltk.corpus import stopwords

setStopWords = set(stopwords.words("english"))
import re

import matplotlib

if execute_on_ssh_compute:
    # see
    # <https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable>
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (7.55, 4.0)
plt.rcParams["axes.titlesize"] = "small"
plt.rcParams["axes.titleweight"] = "light"

import Frames
from Frames import GenericFrame

logger = loguru.logger


# noinspection PyBroadException
class UserLabelCluster:
    classes: []
    classes_to_one_hot_encode_dict: dict

    def __init__(self, user_labels: List[str], word2vec_dict: dict, cluster_k: int, word2vec_embedding_size=300,
                 semantic_clustering=True, iteration=0):
        """
        Initiates a UserLabelCluster

        :param user_labels: the user label which will be available during training (do NOT input user labels
            = frames which are in validation or even test data!)
        :param word2vec_dict: the embedding dictionary, for example glove
        :param cluster_k: how many clusters di you want to have? Must be lower equal the number of user labels.
            However, there are some special cases:
            <ul>
                <li> 1: only one cluster, so, there is nothing to predict (making wrong)</li>
                <li> -1: clustering is disabled. It's the vanilla version:
                    <code>
                    set_of_user_frames = {sample.get("frame", "n/a")
                    for sample in samples[: int(len(samples) * training_set_percent * 0.95)]}
                    dict_of_user_frames = {frame: num for num, frame in enumerate(set_of_user_frames)}
                    amount_clusters_dict_of_user_frames = max(dict_of_user_frames.values()) + 1
                    </code>
                </ul>
        :param word2vec_embedding_size: the size of each word2vec embedding
        :param semantic_clustering: determines whether a semantic clustering should be enabled.
            <ul>
                <li><code>True</code>: user labels go through a preprocessing:
                removing stopwords, emphasise keywords, ...</li>
                <li><code>False:</code>: plain vanilla user label is used</li>
            </ul>
        """
        assert cluster_k == -1 or 1 <= cluster_k <= len(user_labels)

        self.classes = list()
        self.classes_to_one_hot_encode_dict = dict()
        self.word2vec_dict = word2vec_dict
        self.word2vec_embedding_size = word2vec_embedding_size
        self.cluster_k = cluster_k

        self.semantic_clustering = semantic_clustering

        if self.cluster_k != 1:
            for user_label in user_labels:
                self.insert_class(user_label=user_label)
        else:
            logger.warning("You choose a special case! "
                           "With only 1 big cluster, all data-points will lay in this cluster. "
                           "Hence, there is nothing to compute...")

        if self.cluster_k > 1:
            path = pathlib.Path("clusters", "{}x{}d_{}_{}c_{}.pkl".format(len(user_labels), word2vec_embedding_size,
                                                                          "semantic" if semantic_clustering else "",
                                                                          cluster_k, iteration))
            path.parent.mkdir(exist_ok=True, parents=True)
            if path.exists():
                logger.debug("You computed the clusters already once, here: {}", path.absolute())
                self.cluster, self.classes_to_one_hot_encode_dict = pickle.load(path.open(mode="rb"))
                logger.success("Successfully loaded the already computed cluster: {}", path.name)
            else:
                logger.trace("Compute the cluster now...")
                self.cluster = nltk.cluster.KMeansClusterer(num_means=self.cluster_k,
                                                            distance=nltk.cluster.cosine_distance,
                                                            repeats=10 + int(math.sqrt(self.cluster_k) * 3),
                                                            avoid_empty_clusters=True)
                self.finalize_class_set()
                logger.success("Yes, we will store it in \"{}\"", path.name)
                pickle.dump((self.cluster, self.classes_to_one_hot_encode_dict), path.open(mode="wb"))
                logger.trace("Pickling done: {}", path.stat())
        elif self.cluster_k == -1:
            logger.warning("You disabled the clustering!"
                           "Hence, it's possible that further predictions will lead to outputs like"
                           "\"Your input is in no particular \"class\"\"")
            self.classes_to_one_hot_encode_dict = {f: i for i, f in enumerate(self.classes)}

    def insert_class(self, user_label: str) -> None:
        """
        Inserts a new user label (frame) which should be used by the cluster

        NOT RECOMMENDED TO USE FROM OUTSIDE!

        :param user_label: the user label which should be inserted
        :return: nothing - just updates the internal structure. Has no effect without using self.finalize_class_set
        """
        logger.debug("Adds \"{}\" to the user label class", user_label)

        if self.cluster_k == -1:
            logger.debug("Clustering is disabled. Hence, just added to the list in set-semantic (current length: {})",
                         len(self.classes))
        final_label_tokens = self.convert_label(user_label=user_label)
        if final_label_tokens not in self.classes:
            self.classes.append(final_label_tokens)
        else:
            logger.debug("\"{}\" was already in the list!", " ".join(final_label_tokens))

    def convert_label(self, user_label: str) -> Tuple[str]:
        """
        FOR INTERNAL USE ONLY!

        :param user_label: the user label (frame)
        :return: a converted tokenized Tuple-list for further processing
        """
        user_label = re.sub(string=user_label, pattern="(?<=\w)\/(?=\w)", repl=" ", count=1)
        final_label_tokens = nltk.word_tokenize(text=user_label, language="english", preserve_line=False)
        for i, token in enumerate(final_label_tokens):
            token = token.lower()
            if token == "v" or token == "v." or token == "vs" or token == "vs.":
                final_label_tokens[i] = "versus"
        if self.semantic_clustering:
            tagged_label = [t_tag for t_tag in nltk.pos_tag(tokens=final_label_tokens, lang="eng", tagset="universal")
                            if t_tag[0] not in setStopWords]
            tagged_label.reverse()
            final_label_tokens = [t_tag[0] for t_tag in tagged_label if t_tag[1] == "NOUN"] + \
                                 [t_tag[0] for t_tag in tagged_label if t_tag[1] == "VERB"] + \
                                 [t_tag[0] for t_tag in tagged_label if t_tag[1] not in ["NOUN", "VERB"]]
        logger.debug("Converted the user label \"{}\" to \"{}\"", user_label, " ".join(final_label_tokens))
        if len(final_label_tokens) > 4:
            logger.warning("The label {} has more than 4 tokens: {}. Discard {}", final_label_tokens,
                           len(final_label_tokens), final_label_tokens[4:])
            final_label_tokens = final_label_tokens[:4]
        elif len(final_label_tokens) == 0:
            logger.warning("We receive an preprocessed user label which is empty!")
            final_label_tokens = ["<pad>"] * 4
        elif len(final_label_tokens) == 1:
            final_label_tokens = final_label_tokens * 4
        elif len(final_label_tokens) == 2:
            final_label_tokens = final_label_tokens * 2
        elif len(final_label_tokens) == 3:
            final_label_tokens.append(final_label_tokens[0])

        return tuple(final_label_tokens)

    def finalize_class_set(self) -> None:
        """
        UPDATES THE INTERNAL STRUCTURE

        :return: nothing
        """
        logger.info("We have {} distinct classes, let's cluster it!", len(self.classes))

        logger.debug("Created a cluster instance {} and this will cluster {} samples", self.cluster, self.classes)
        try:
            assigned_clusters = self.cluster.cluster(vectors=[self.convert_str_list_to_vector(c) for c in self.classes],
                                                     assign_clusters=True, trace=not execute_on_ssh_compute)
        except Exception:
            logger.exception("Failed to cluster the actual class set ({} samples)", len(self.classes))
            return

        self.classes_to_one_hot_encode_dict.clear()
        for i in range(len(self.classes)):
            self.classes_to_one_hot_encode_dict[self.classes[i]] = assigned_clusters[i]

    def convert_str_list_to_vector(self, string_list: Tuple[str]) -> numpy.ndarray:
        """
        FOR INTERNAL USE ONLY!

        :param string_list: a tuple list of tokens. Must be exactly 4
        :return: a one-dimensional (concatenated) numpy-array. See word embeddings
        """
        if len(string_list) != 4:
            logger.error("convert_str_list_to_vector got a too short or long string list: {}. We return a zero-vector!",
                         string_list)
            return numpy.zeros(shape=(self.word2vec_embedding_size +
                                      self.word2vec_embedding_size / 2 +
                                      self.word2vec_embedding_size / 3 +
                                      self.word2vec_embedding_size / 4,),
                               dtype="float32"
                               )
        ret = numpy.zeros(shape=(0,), dtype="float32")
        for i, token in enumerate(string_list):
            logger.trace("Process the {}. token \"{}\"", (i + 1), string_list[i])
            ret = numpy.concatenate([ret,
                                     numpy.average(
                                         numpy.reshape(
                                             self.word2vec_dict.get(string_list[i],
                                                                    numpy.negative(
                                                                        numpy.ones(
                                                                            shape=(self.word2vec_embedding_size,),
                                                                            dtype="float32")
                                                                    )),
                                             (int(self.word2vec_embedding_size / (i + 1)), (i + 1))
                                         ),
                                         axis=1)],
                                    axis=0)
        return ret

    def get_y(self, user_label: str) -> numpy.ndarray:
        """
        Gets the ground truth one-hot-encoded label for the particular user label

        :param user_label: a user label (frame)
        :type user_label: str
        :return: an numpy array
        """
        final_user_label = self.convert_label(user_label=user_label)
        if self.cluster_k == -1:
            index = self.classes_to_one_hot_encode_dict.get(final_user_label, len(self.classes_to_one_hot_encode_dict))
            ret = numpy.zeros(shape=(len(self.classes_to_one_hot_encode_dict) + 1,), dtype="float32")
            ret[index] = 1.
            return ret
        elif self.cluster_k == 1:
            return numpy.ones(shape=(1,), dtype="float32")

        if final_user_label in self.classes_to_one_hot_encode_dict.keys():
            cluster_index = self.classes_to_one_hot_encode_dict[final_user_label]
        else:
            logger.info("We never saw the converted user_label \"{}\" - predict the cluster for it!",
                        " ".join(final_user_label))
            cluster_index = self.cluster.classify(vector=self.convert_str_list_to_vector(final_user_label))
            logger.debug("The cluster index of {} is {} - add it to the dictionary!", final_user_label, cluster_index)
            self.classes_to_one_hot_encode_dict[final_user_label] = cluster_index

        ret = numpy.zeros(shape=(self.cluster_k,), dtype="float32")
        ret[cluster_index] = 1.
        return ret

    def get_y_length(self) -> int:
        """
        :return: the length of a returned vector by self.get_y
        """
        if self.cluster_k == -1:
            return len(self.classes_to_one_hot_encode_dict) + 1

        return self.cluster_k

    def __str__(self) -> str:
        return "{}{}cluster_{}z{}".format("Semantic" if self.semantic_clustering else "",
                                          "-{}d-".format(self.word2vec_embedding_size)
                                          if self.word2vec_embedding_size != 300 else "",
                                          len(self.classes),
                                          self.get_y_length())


def load_csv(data_set: str, frames: Frames, filter_unknown_frames=True, shuffle_samples=True, under_sampling=False,
             limit_data=-1) -> List[dict]:
    logger.info("Read data set at {}", os.path.abspath(data_set))
    data = []
    with open(data_set, newline="\n", encoding="utf-8") as csv_file:
        csvReader = csv.reader(csv_file, delimiter="|", quotechar='"')
        scheme = None
        for row in csvReader:
            if csvReader.line_num == 1:
                scheme = row
            else:
                if scheme is None:
                    logger.error("No scheme!")
                    raise AttributeError
                else:
                    logger.debug("Fetch {}", ', '.join(row))
                    argumentMapping = dict()
                    for i in range(0, len(row)):
                        try:
                            argumentMapping[scheme[i]] = row[i]
                        except IndexError as err:
                            logger.warning(err)
                    logger.debug("Collected {} elements.", len(argumentMapping))
                    data.append(argumentMapping)

    if limit_data >= 1:
        logger.warning("You want to limit your test data! You want to scale it down to {}", limit_data)
        data = data[:limit_data]
        logger.info("Took the first {}", len(data))
        if under_sampling:
            logger.warning("Beware, that the current number of samples ({}) will reduce probably"
                           "by the activated under-sampling!", len(data))

    if shuffle_samples:
        random.shuffle(data)

    if under_sampling:
        distribution = {fn: [] for fn in frames.frame_names}
        if not filter_unknown_frames:
            distribution["__UNKNOWN__"] = []

        for sample in data:
            frame = frames.map_user_label_to_generic(sample.get("frame", "FETCH_ERROR"))[0][0]
            try:
                distribution[frame] += [sample]
            except KeyError as e:
                logger.debug("Exception {}", type(e))
                if filter_unknown_frames:
                    logger.trace("Unexpected frame: {}", frame)
                else:
                    logger.error("Unexpected frame: {}", frame)

        min_frame = min(distribution.items(), key=lambda item: len(item[1]))

        if len(min_frame[1]) <= 0:
            logger.critical(
                "You activated the under-sampling, but frame {} has zero members... Hence, you discard all!",
                min_frame[0])
            exit(1)
        else:
            logger.info("Under-sample now... key point is {}", min_frame)

        data = []
        for frame, samples in distribution.items():
            logger.debug("Gather {} samples now for frame \"{}\"", min_frame[1], frame)
            data.extend(samples[:len(min_frame[1])])

        logger.warning("Your data is reduced to {} samples by under-sampling", len(data))

    if shuffle_samples:
        random.shuffle(data)

    return data


def load_word_embeddings(glove_file: pathlib.Path, embedding_size: int) -> dict:
    logger.info("Load word embeddings from \"{}\" ({}d)", glove_file.absolute(), embedding_size)

    word_vector_map = dict()
    if glove_file.exists():
        with glove_file.open(mode="r", encoding="utf-8") as reader:
            for line in reader:
                word_vector_map[line[:line.index(" ")].strip()] = numpy.fromstring(line[line.index(" "):].strip(),
                                                                                   dtype="float32", sep=" ",
                                                                                   count=embedding_size)
            logger.info("Collected {} word embeddings", len(word_vector_map))
    else:
        logger.critical("Either \"{}\" doesn't exists in general or you forgot to downloaded it!",
                        glove_file.absolute())
        exit(-10)

    logger.debug("Loaded {} word embeddings", len(word_vector_map))
    return word_vector_map


def prepare_X(arguments: List[dict], max_seq_len: int, word_embedding_dict: dict, word_embedding_length: int,
              filter_unknown_frames=False, frame_set=Frames.media_frames_set, using_topic=False, using_premise=True,
              using_conclusion=True) -> numpy.ndarray:
    assert isinstance(arguments, list)
    num_samples = len(arguments)
    if filter_unknown_frames:
        arguments = [s for s in arguments if s.get("genericFrame", "__UNKNOWN__") in frame_set.frame_names]
        logger.info("You ignore unknown frames. This costs you in this section {} samples",
                    (num_samples - len(arguments)))
        num_samples = len(arguments)
    ret = numpy.zeros(shape=(num_samples, max_seq_len, word_embedding_length), dtype="float32")

    logger.debug("Created a y-output-matrix of shape {}", ret.shape)

    for c_r, sample in enumerate(arguments, 0):
        arg_string = reduce(lambda arg1, arg2: arg1 + " " + arg2,
                            argument_to_str(sample, using_topic=using_topic,
                                            using_premise=using_premise, using_conclusion=using_conclusion))
        logger.trace("Processes Arg[{}] now", arg_string)
        tokens = [t.lower() for t in nltk.word_tokenize(arg_string)]
        if len(tokens) > max_seq_len:
            if len(tokens) > max_seq_len * 2:
                logger.warning("\"{}\" has more then {} tokens: {}. Consider a longer max_length!", arg_string,
                               max_seq_len, len(tokens))
            else:
                logger.debug("\"{}\" has more then {} tokens: {}", arg_string, max_seq_len, len(tokens))

            tokens = tokens[:max_seq_len]

        for t_r, token in enumerate(tokens, 0):
            ret[c_r, t_r] = word_embedding_dict.get(token, numpy.ones(shape=(word_embedding_length,)))

    logger.debug("Pre-Processed {} X-values now", len(arguments))

    return ret


def compute_y_frame_distribution(samples: List[Dict], frames: Union[GenericFrame, UserLabelCluster],
                                 ignore_unknown=False, enable_fuzzy_framing=False) -> numpy.ndarray:
    num_samples = len(samples)
    if ignore_unknown and isinstance(frames, GenericFrame):
        samples = [s for s in samples if s.get("genericFrame", "__UNKNOWN__") in frames.frame_names]
        logger.info("You ignore unknown frames. This costs you in this section {} samples",
                    (num_samples - len(samples)))
        num_samples = len(samples)
    ret = numpy.zeros(shape=(num_samples,
                             frames.get_prediction_vector_length(ignore_unknown=ignore_unknown)
                             if isinstance(frames, GenericFrame) else frames.get_y_length()
                             ),
                      dtype="float32")

    logger.debug("Created a y-output-matrix of shape {}", ret.shape)

    for c_r, sample in enumerate(samples, 0):
        if isinstance(frames, GenericFrame):
            if not enable_fuzzy_framing:
                ret[c_r] = frames.decode_frame_label(sample.get("genericFrame", "__UNKNOWN__"),
                                                     ignore_unknown=ignore_unknown)
            else:
                frame_distribution = [frame.strip("() ").split(":", 2) for frame in
                                      str(sample.get("fuzzyFrame", "(__UNKNOWN__:1.0)")).split(") (")]
                try:
                    frame_distribution = {f_d[0]: float(f_d[1]) for f_d in frame_distribution}
                    ret[c_r] = frames.decode_frame_label(frame_distribution, ignore_unknown=ignore_unknown)
                except ValueError as e:
                    logger.error("Failure {}: leave the {}. prediction vector blank!", e, c_r)
        elif isinstance(frames, UserLabelCluster):
            ret[c_r] = frames.get_y(user_label=sample.get("frame", "neutral"))

    logger.trace("DONE")

    return ret


def compute_y_user_label_to_generic_frame_distribution(samples: List[Dict], word2vec: Dict, frames: GenericFrame,
                                                       enable_fuzzy_framing=False,
                                                       enable_other_class=True) -> numpy.ndarray:
    num_samples = len(samples)

    model = word_mover_distance.WordEmbedding(model=word2vec)
    logger.trace("Created a word_mover_distance-model: {}", model)
    frames_tokens = frames.get_all_frame_names(tokenized=True, lower=True)
    logger.debug("We will compute the distances to the following frames: {}", ", ".join(frames.frame_names))

    ret = numpy.zeros(shape=(num_samples,
                             frames.get_prediction_vector_length(ignore_unknown=not enable_other_class)),
                      dtype="float32")
    logger.debug("Created a return-template of shape {}", ret.shape)
    for i, sample in enumerate(samples):
        frame = re.sub(string=sample.get("frame", "unknown").strip("\"' "),
                       pattern="(?<=\w)\/(?=\w)", repl=" ", count=1)
        logger.trace("Fetched a label (user frame): {}", frame)
        frame_tokens = [t.lower() for t in nltk.word_tokenize(text=frame, language="english", preserve_line=False)]
        logger.trace("Will compute the word-movers-distance to [{}]", "-".join(frame_tokens))

        word_movers_distances = numpy.zeros(shape=(frames.get_prediction_vector_length(ignore_unknown=not enable_other_class),),
                                            dtype="float32")

        for j, generic_frame_tokens in enumerate(frames_tokens):
            word_movers_distances[j] = model.wmdistance(document1=frame_tokens, document2=generic_frame_tokens)
        if enable_other_class:
            word_movers_distances[-1] = (numpy.max(word_movers_distances)-numpy.min(word_movers_distances[:-1])) *\
                                        word_movers_distances.shape[0] * 0.5

        logger.trace("Total distribution: {} (not normalized)", word_movers_distances)
        if enable_fuzzy_framing:
            word_movers_closeness = numpy.add(numpy.negative(word_movers_distances), numpy.max(word_movers_distances))
            ret[i] = numpy.divide(word_movers_closeness, max(numpy.array(0.001, dtype="float32"),
                                                             numpy.sum(word_movers_closeness)))
        else:
            min_index = 0
            min_distance = word_movers_distances[0]
            for j, d in enumerate(word_movers_distances):
                if d < min_distance:
                    min_index = j
                    min_distance = d

            ret[i, min_index] = 1.0

    return ret


def compute_y_word_embedding(samples: List[dict], word_vector_map: dict, embedding_length: int, filter_stop_words=True,
                             max_seq_len=-1) -> numpy.ndarray:
    assert isinstance(samples, list)
    assert isinstance(word_vector_map, dict)
    num_samples = len(samples)

    if max_seq_len <= 0:
        ret = numpy.zeros(shape=(num_samples, embedding_length), dtype="float32")
    else:
        ret = numpy.zeros(shape=(num_samples, max_seq_len, embedding_length), dtype="float32")

    logger.debug("Created a y-output-matrix of shape {}", ret.shape)

    for c_r, sample in enumerate(samples, 0):
        frame = sample.get("frame", None)
        if frame is None:
            logger.warning("You sample ({}) doesn't provide a [user] frame label -  maybe you sent a wrong .csv-file "
                           "to this, but you have to use the basic csv-file!", sample)
        else:
            logger.trace("Process the frame \"{}\" now - first the NLP-basic pipeline", frame)
            frame = re.sub("(?!-)\W", " ", frame)
            frame = frame.strip()
            frame = re.sub("\d+", "number", frame)
            frame = re.sub("'\w*\s", " ", frame)
            tokens = [t for t in nltk.word_tokenize(frame) if not filter_stop_words or t not in setStopWords]
            logger.debug("Pre-processing done: \"{}\" -> {}", frame, tokens)

            if len(tokens) == 0:
                logger.warning("Strange - the user label \"{}\" (clean: \"{}\") does not contain any token!",
                               sample["frame"], frame)
            else:
                if max_seq_len <= 0:
                    embedding_list = [word_vector_map.get(t, numpy.zeros(shape=(embedding_length,))) for t in tokens]
                    final_embedding = numpy.average(embedding_list, axis=0)
                else:
                    # the cosine similarity of zeros is always 0 (lowest), no matter what we predict.
                    # This isn't our goal, hence, we decide for
                    # 1-vector: unknown, but existing token
                    # -1-vector: not existing token = padding token
                    embedding_list = [word_vector_map.get(t, numpy.ones(shape=(embedding_length,))) for t in tokens]
                    final_embedding = embedding_list[:max_seq_len] \
                        if len(embedding_list) >= max_seq_len else \
                        (embedding_list +
                         [numpy.negative(numpy.ones(shape=(embedding_length,)))]
                         * (max_seq_len - len(embedding_list)))

                logger.trace("Final embedding is: {}", final_embedding)

                ret[c_r] = final_embedding

    return ret


def argument_to_str(input_argument_mapping, using_topic=False, using_premise=True, using_conclusion=True) -> List[str]:
    try:
        output_argument_list = []
        if using_topic:
            output_argument_list.append(input_argument_mapping.get("topic", "<unk>"))
        if using_premise:
            output_argument_list.append(input_argument_mapping.get("premise", "<unk>"))
        if using_conclusion:
            output_argument_list.append(input_argument_mapping.get("conclusion", "<unk>"))
        return output_argument_list
    except AttributeError as e:
        logger.error("Error {}: the input {} is no (valid) map - maybe you projected the input already?", e,
                     input_argument_mapping)


# noinspection PyBroadException
def save_model(model: keras.Model, model_save_path=None, additional_metrics_to_plot=None) -> pathlib.Path:
    if additional_metrics_to_plot is None:
        additional_metrics_to_plot = []
    if model_save_path is None:
        logger.warning("Utils.save_model receives no model_save_path... try to construct one")
        model_save_path = pathlib.Path("trained_model").joinpath(
            "{}-{}".format(model.name, round(datetime.datetime.now().timestamp()))
        )
    try:
        model.save(filepath=(model_save_path.absolute()), overwrite=True, save_format="tf")
        logger.info("Save the trained model now in \"{}\"", model_save_path.absolute())
    except Exception as e:
        logger.error("Fail to save the fine-tuned NN in \"{}\", because of {}", model_save_path.absolute(),
                     type(e))
        try:
            pickle.dump(model, pathlib.Path("{}.pkl".format(model_save_path)).open(mode="wb"))
        except Exception:
            logger.error("Failed also to pickle the fine-tuned NN in \"{}\" - give it up...",
                         "{}.pkl".format(model_save_path))

    try:
        if model.history is not None:
            logger.info("Interesting, \"{}\" has a history... save it!", model.name)
            # Plot history
            plt.plot(model.history.history["loss"], label='loss (train)')
            plt.plot(model.history.history["val_loss"], label='loss (val)')
            if len(additional_metrics_to_plot) == 0:
                train_acc = "categorical_accuracy" if "categorical_accuracy" in model.history.history.keys() else \
                    ("accuracy" if "accuracy" in model.history.history.keys() else None)
                val_acc = "val_categorical_accuracy" if "val_categorical_accuracy" in model.history.history.keys() else \
                    ("val_accuracy" if "val_accuracy" in model.history.history.keys() else None)
                try:
                    if train_acc is not None:
                        plt.plot(model.history.history[train_acc], label="categorical accuracy (train)")
                    if val_acc is not None:
                        plt.plot(model.history.history[val_acc], label="categorical accuracy (val)")
                except KeyError as e:
                    logger.error(e)
                    train_acc = None
                    val_acc = None
                if train_acc is not None and val_acc is not None:
                    plt.suptitle(t="Results of {}".format(model.name), fontsize="large", fontweight="demi")
                    plt.title("top-acc-train: {}/ top-acc-val: {}".format(
                        round(max(model.history.history[train_acc]), 3),
                        round(max(model.history.history[val_acc]), 3)))
                else:
                    plt.suptitle(t="Results of {}".format(model.name), fontsize="large", fontweight="book")
                    plt.title("top-acc-train: {}/ top-acc-val: {}".format(
                        round(min(model.history.history["loss"]), 3),
                        round(min(model.history.history["val_loss"]), 3)))
            else:
                for metric in additional_metrics_to_plot:
                    if metric in model.history.history.keys():
                        plt.plot(model.history.history[metric], label=metric)
                    else:
                        logger.error(
                            "You want to plot the metric \"{}\", but its not available in the history, only {}",
                            metric, ", ".join(model.history.history.keys()))
                title_appendix_list = ["{}:{}".format(m, round(max(model.history.history[m]), 3))
                                       for m in additional_metrics_to_plot if m in model.history.history.keys()]
                try:
                    plt.suptitle(t="Results of {}".format(model.name), fontsize="large", fontweight="roman")
                    plt.title("{}".format("|".join(title_appendix_list[:min(2, len(title_appendix_list))])
                                          if len(title_appendix_list) >= 1 else round(
                        min(model.history.history["val_loss"]), 3)))
                except KeyError as e:
                    logger.warning("Failure in plot (title): {}", e)
                    plt.title("Plot of {}".format(model.name))
            plt.ylabel('loss/ accuracy')
            plt.xlabel('No. epoch')
            plt.legend(loc="upper left")
            plt.grid(b=True, which="major", axis="y", color="gray", alpha=0.5, animated=False, linestyle="-",
                     linewidth=1.0)
            model_save_path_plot = model_save_path.joinpath("plot.png")
            logger.info("Will save plot to {}", model_save_path_plot)
            plt.savefig(fname=str(model_save_path_plot.absolute()), transparent=True)
    except Exception:
        logger.exception("Failed to save the plot")
    finally:
        return model_save_path


def add_plot_description(additional_text: str, model_save_path: pathlib.Path) -> None:
    if additional_text is not None:
        remain_text = additional_text
        print_text = ""
        while len(remain_text) > 0:
            print_text += "{}\n".format(remain_text[:min(len(remain_text), 40)])
            remain_text = "" if len(remain_text) <= 40 else remain_text[40:]
        logger.trace("You want to print additional text on this plot: \"{}\". OK, I will try it!",
                     additional_text)
        plt.figtext(x=0.15, y=0.15, s=print_text)
    model_save_path_plot = model_save_path.joinpath("plot.png")
    logger.info("Will save plot to {}", model_save_path_plot)
    plt.savefig(fname=str(model_save_path_plot.absolute()), transparent=True)


def load_pre_trained_model(path: Optional[pathlib.Path]) -> Optional[keras.Model]:
    if path is None:
        return None
    if path.exists():
        try:
            return keras.models.load_model(path)
        except ImportError as e:
            logger.critical("Your given file path \"{}\" don't contain a valid h5-saved model: {}", path.name, e)
        except IOError as e:
            logger.error("IO-Error: {} (at \"{}\") - can't load the model!", e, path.absolute())
    else:
        logger.error("Your given path \"{}\" does not exists - no model load!", path.absolute())

    return None


def to_same_sample_amount(data_lists: List[List[Dict]], under_sampling=False) -> List[List[Dict]]:
    logger.info("You want the same number of samples in corpora [{}]",
                ", ".join(map(lambda cor: "{} samples".format(len(cor)), data_lists)))

    if under_sampling:
        min_length = min(data_lists, key=lambda c: len(c))
        min_length = len(min_length)
        logger.warning("You under-sample! This means, that you will throw data away! Shrink to {}", min_length)
        return [cor[:min_length] for cor in data_lists]
    else:
        logger.debug("Extend the shorter corpora")
        len_data_lists = [len(cor) for cor in data_lists]
        if all(map(lambda c: c == len_data_lists[0], len_data_lists)):
            logger.info("All of your corpora are already equal in the number of samples! Nothing to do!")
            return data_lists
        else:
            max_length = max(len_data_lists)
            logger.info("Extend all corpora to a size of {}", max_length)
            return [extend_corpus_to_size(cor, max_length) for cor in data_lists]


def extend_corpus_to_size(corpus: List[Dict], size: int) -> List[Dict]:
    if len(corpus) >= size:
        logger.debug("Nothing to do. The corpus has already the size {}", len(corpus))
        return corpus
    if len(corpus) == 0:
        logger.critical("The input corpus have not to be empty!")
        return []

    ret = []
    while len(ret) < size:
        corpus = corpus.copy()
        random.shuffle(corpus)
        if len(ret) + len(corpus) <= size:
            logger.trace("Extended the return list by the full batch of the input list")
            ret.extend(corpus)
        else:
            missing = size - len(ret)
            logger.trace("Extended the return list by the partial batch of the input list ({})", missing)
            ret.extend(corpus[:missing])

    return ret


def return_user_label_specific_word2vec_embedding(word2vec_dict: dict, train_user_labels: [dict],
                                                  embedding_length=None) -> dict:
    if embedding_length is None:
        embedding_length = [i for i in word2vec_dict.values()][0]
        if len(embedding_length) == 2:
            embedding_length = len(embedding_length[1])
        else:
            embedding_length = len(embedding_length)

    ret = dict()

    for user_label in train_user_labels:
        user_label = user_label.get("frame", "")
        user_label = [word for word in nltk.word_tokenize(text=user_label, language="english") if
                      word not in setStopWords]

        for word in user_label:
            if word in ret.keys():
                ret[word] = (ret[word][0] + 0, ret[word][1])
            else:
                word_vector = word2vec_dict.get(word, numpy.ones(shape=(embedding_length,), dtype="float32"))
                ret[word] = (1, word_vector)

    ret["<padding>"] = (int(len(train_user_labels) / 2),
                        numpy.negative(numpy.ones(shape=embedding_length, dtype="float32")))

    return ret


def calculates_predicted_words_specific_frame(word_vectors_prediction: Union[tensorflow.Tensor, numpy.ndarray],
                                              target_word2vec: dict,
                                              embedding_size=None, output_vector_tensors=False) -> \
        [(str, Union[tensorflow.Tensor, numpy.ndarray])]:
    target_word2vec_exclude_padding = target_word2vec.copy()
    if "<padding>" in target_word2vec:
        sample_value = target_word2vec_exclude_padding.pop("<padding>")
    else:
        sample_key, sample_value = target_word2vec_exclude_padding.popitem()
        logger.warning("Your target dictionary ({} keys) doesn't contain the padding-token! Popped \"{}\" instead",
                       len(target_word2vec), sample_key)
    sample_value_is_tuple = True
    if len(sample_value) != 2:
        sample_value_is_tuple = False
        target_word2vec = {k: (1, v) for k, v in target_word2vec.items()}
        target_word2vec_exclude_padding = {k: (1, v) for k, v in target_word2vec_exclude_padding.items()}
    if embedding_size is None:
        logger.debug("We must calculate the embedding size first")
        embedding_size = len(sample_value[1] if sample_value_is_tuple else sample_value)
        logger.trace("The embedding size is {}", embedding_size)

    if len(word_vectors_prediction.shape) == 3:
        logger.warning("You consider to input a batch into this function. "
                       "This function us for single predictions. However, we'll handle this")
        return [calculates_predicted_words_specific_frame(word_vectors_prediction=single_prediction,
                                                          target_word2vec=target_word2vec,
                                                          embedding_size=embedding_size) for single_prediction in
                word_vectors_prediction]
    elif len(word_vectors_prediction.shape) not in [2, 3]:
        logger.error("This function expects a input of shape (word_vector_number, word_vector_vector), "
                     "but you input a shape of {}!", word_vectors_prediction.shape)
        return []
    elif word_vectors_prediction.shape[-1] != embedding_size:
        logger.error("Expects the same word embedding size, but get as prediction {}d und as dictionary {}d",
                     word_vectors_prediction.shape[-1], embedding_size)
        return []

    ret = []
    first_token_process = True
    if isinstance(word_vectors_prediction, numpy.ndarray):
        for word_vector in word_vectors_prediction:
            logger.trace("Calculates the cosine similarity with respect to the word frequency."
                         "Source: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists")
            cal = [(target_word, target_data[1],
                    (-1 + numpy.dot(word_vector, target_data[1]) /
                     (numpy.linalg.norm(word_vector) * numpy.linalg.norm(target_data[1]))) *
                    (0.7 + (1 / max(1, target_data[0]) * 0.3))) for target_word, target_data
                   in (target_word2vec_exclude_padding.items() if first_token_process else target_word2vec.items())]
            ret.append(max(cal, key=lambda c: c[2])[:2])
            logger.trace("Selected from {} the element {}", cal, ret[-1])
    elif isinstance(word_vectors_prediction, tensorflow.Tensor):
        def cos(fn_word_vector):
            nonlocal first_token_process
            logger.trace("Calculates the cosine similarity in tensorflow with respect to the word frequency."
                         "Source: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists")
            fn_cal = [(target_word, target_data[1],
                       tensorflow.multiply(
                           tensorflow.subtract(
                               tensorflow.divide(
                                   tensorflow.reduce_sum(tensorflow.multiply(fn_word_vector, target_data[1])),
                                   tensorflow.multiply(tensorflow.norm(fn_word_vector),
                                                       numpy.linalg.norm(target_data[1]))),
                               1),
                           (0.7 + (1 / max(1, target_data[0]) * 0.3))
                       )
                       ) for target_word, target_data
                      in (target_word2vec_exclude_padding.items() if first_token_process else target_word2vec.items())]

            first_token_process = False
            max_elem = fn_cal[0]
            if len(fn_cal) >= 2:
                for c in fn_cal[1:]:
                    if tensorflow.reduce_all(tensorflow.math.greater_equal(c[2], max_elem[2])):
                        max_elem = c
            ret.append(max_elem[:2])
            return max_elem[1]

        first_token_process = True
        if output_vector_tensors:
            return tensorflow.map_fn(fn=cos, elems=word_vectors_prediction)
        else:
            tensorflow.map_fn(fn=cos, elems=word_vectors_prediction)
        logger.debug(
            "Calculated for the current word-vector-prediction the corresponding word \"{}\" out of {} choices",
            ret[-1], len(target_word2vec))
    logger.info("Finished the mapping process ({} tokens): {}", len(ret), ret)

    return ret
