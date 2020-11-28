"""
Beware of the Params-section right below the imports!

- general in- and output of the NN
  - Inputs / Shapes/ Preprocessing
    - max_seq_len
    - what to use from the plain input
      - using_topic
      - using_premise
      - using_conclusion
      - word_embeddings + embedding_size
      - word_embeddings_output: with having a number here + using_word_embedding_output = True,
      you activate the mode of predicting word embeddings (requires embedding_size_output)
      - frames: the Frames.GenericFrame-set mainly for the Media.Frames-task-prediction
      - one_hot_output_clusters:
        - with having a number here, a K-MeansClusterer will cluster all user labels for the Webis-task ==> one-hot-
        encoding
        - with having a Frames.GenericFrame-set here, the word-embedding next to a generic frame will be predicted
        (without fuzzy framing)
- Settings for the NN (architecture)
  - NN_which_used
  - batch_size
  - epochs
  - iterations
- regarding the corpora
  Besides the data sets
  - under_sampling: reduces IN a dataset to that class which have the lowest coverage
  - shuffle_samples: to shuffle the samples
  - and the parts of training, validation and test in %/100 (so for the half write 0.5)
"""

from typing import Union

# We executed the code on strong CPU clusters without an GPU (ssh compute). Because of this extraordinary
# executing environment, we introduce this flag. To reproduce the results in the paper, enable this flag.
execute_on_ssh_compute = False

import pathlib
import sys

import loguru
import nltk
import tensorflow.keras as keras
import tensorflow.keras.layers as keras_layers

from Metric_UserLabel import UserLabelPredictionMetric

logger = loguru.logger
logger.remove()
logger.add(sink=sys.stdout, level="INFO", colorize=True, backtrace=True, catch=True)

import Frames
import Utils

#####################################
# ### PARAMETERS ####################
#####################################
max_seq_len = 500  # very important! Best situation is when no sample is clipped (default = 500)
enable_fuzzy_framing = False
using_topic = False
using_premise = True
using_conclusion = True
filter_unknown_frames = False
word_embeddings_path_base = pathlib.Path.home().joinpath("remote", "glove") \
    if execute_on_ssh_compute else pathlib.Path.home().joinpath("Documents", "Einsortiertes",
                                                                "Nach Lebensabschnitten einsortiert",
                                                                "Promotionszeit (2019-2023)", "Promotion",
                                                                "Programming", "_wordEmbeddings", "glove")
word_embeddings = word_embeddings_path_base.joinpath("glove.840B.300d.txt")
embedding_size = 300

using_word_embedding_output = False
word_embeddings_output = word_embeddings_path_base.joinpath("glove.6B.50d.txt")
embedding_size_output = 50
max_seq_len_output = 2
frames = None  # if None, the one-hot-encoded embedded frame is predicted
one_hot_output_clusters: Union[int, Frames.GenericFrame]
one_hot_output_clusters = 25

# Training - fit
trained_model = None
# trained_model = pathlib.Path(
#     "trained_model",
#     "FrameTask",
#     "BiGRU",
#     "Webis-argument-framing_out-500-glove.840B.300d-MediaFramesSet+1-0010--premise-conclusion"
# )
NN_which_used = "BiGRU"  # "BiGRU", "BiLSTM" or "CNN" - Naderi & Hirst showed better results with GRUs
assert NN_which_used in ["BiGRU", "BiLSTM", "CNN"]
batch_size = 64
epochs = 12
iterations = 12

# Dataset
data_set = pathlib.Path("Corpora", "Webis-argument-framing-mediaFramesTopics_out.csv")
# data_set = pathlib.Path("Corpora").joinpath("MediaFramesCorpus").joinpath("converted").joinpath(
#   "Media-immigrationsamesexsmoking-framing_dirty_all_exact_out.csv")
#    "Media-immigrationsamesexsmoking-framing_gold_pure+exact_out.csv")
shuffle_samples = False
training_set_percent = 0.8
validating_set_percent = 0.1
test_set_percent = 0.1
under_sampling = False

#######################################################################################################################

nltk.download("punkt")

#######################################################################################################################

if __name__ == "__main__":
    samples = Utils.load_csv(data_set=data_set, frames=frames, filter_unknown_frames=filter_unknown_frames,
                             shuffle_samples=shuffle_samples and frames is not None, under_sampling=under_sampling)

    word_vector_map = Utils.load_word_embeddings(glove_file=word_embeddings, embedding_size=embedding_size)
    if using_word_embedding_output:
        if word_embeddings != word_embeddings_output:
            word_vector_map_output = Utils.load_word_embeddings(glove_file=word_embeddings_output,
                                                                embedding_size=embedding_size_output)
        else:
            word_vector_map_output = word_vector_map

    model = Utils.load_pre_trained_model(path=trained_model)
    if model is not None:
        iterations = -1

    losses = []
    accuracies = []

    clusters = None
    model_save_path = None
    training_set = None
    test_set = None
    for i in range(1, max(2, iterations + 1)):
        logger.info("Iteration {}", i)
        clusters = None
        if frames is None and not using_word_embedding_output and isinstance(one_hot_output_clusters, int):
            clusters = \
                Utils.UserLabelCluster(user_labels=[sample.get("frame", "n/a")
                                                    for sample in
                                                    samples[: int(len(samples) * training_set_percent)]],
                                       word2vec_dict=word_vector_map,
                                       cluster_k=one_hot_output_clusters,
                                       word2vec_embedding_size=300,
                                       semantic_clustering=True,
                                       iteration=i
                                       )
            logger.info("Found {} distinct frames in test set. That are {} clusters",
                        len(clusters.classes), clusters.get_y_length())
        training_set = None
        if iterations >= 1:
            training_set = samples[:int(len(samples) * training_set_percent)]
            logger.info("Training set contains {} samples", len(training_set))
            training_set_X = Utils.prepare_X(arguments=training_set, word_embedding_dict=word_vector_map,
                                             word_embedding_length=embedding_size, max_seq_len=max_seq_len,
                                             frame_set=frames, filter_unknown_frames=filter_unknown_frames,
                                             using_topic=using_topic, using_premise=using_premise,
                                             using_conclusion=using_conclusion)
            if using_word_embedding_output:
                # noinspection PyUnboundLocalVariable
                training_set_Y = Utils.compute_y_word_embedding(samples=training_set,
                                                                word_vector_map=word_vector_map_output,
                                                                embedding_length=embedding_size_output,
                                                                max_seq_len=max_seq_len_output)
            elif isinstance(one_hot_output_clusters, Frames.GenericFrame) and frames is None:
                training_set_Y =\
                    Utils.compute_y_user_label_to_generic_frame_distribution(samples=training_set,
                                                                             word2vec=word_vector_map,
                                                                             frames=one_hot_output_clusters,
                                                                             enable_fuzzy_framing=enable_fuzzy_framing,
                                                                             enable_other_class=one_hot_output_clusters == Frames.media_frames_set)
            else:
                training_set_Y = Utils.compute_y_frame_distribution(samples=training_set,
                                                                    frames=frames
                                                                    if frames is not None else clusters,
                                                                    enable_fuzzy_framing=enable_fuzzy_framing,
                                                                    ignore_unknown=filter_unknown_frames)

            val_set = samples[len(training_set):len(training_set) + int(len(samples) * validating_set_percent)]
            logger.info("Validation set contains {} samples", len(val_set))
            val_set_X = Utils.prepare_X(arguments=val_set, word_embedding_dict=word_vector_map,
                                        word_embedding_length=embedding_size, max_seq_len=max_seq_len, frame_set=frames,
                                        filter_unknown_frames=filter_unknown_frames, using_topic=using_topic,
                                        using_premise=using_premise, using_conclusion=using_conclusion)
            if using_word_embedding_output:
                val_set_Y = Utils.compute_y_word_embedding(samples=val_set, word_vector_map=word_vector_map_output,
                                                           embedding_length=embedding_size_output,
                                                           max_seq_len=max_seq_len_output)
            elif isinstance(one_hot_output_clusters, Frames.GenericFrame) and frames is None:
                val_set_Y =\
                    Utils.compute_y_user_label_to_generic_frame_distribution(samples=val_set,
                                                                             word2vec=word_vector_map,
                                                                             frames=one_hot_output_clusters,
                                                                             enable_fuzzy_framing=enable_fuzzy_framing,
                                                                             enable_other_class=one_hot_output_clusters == Frames.media_frames_set)
            else:
                val_set_Y = Utils.compute_y_frame_distribution(samples=val_set,
                                                               frames=frames if frames is not None else clusters,
                                                               enable_fuzzy_framing=enable_fuzzy_framing,
                                                               ignore_unknown=filter_unknown_frames)

            test_set = samples[len(training_set) + len(val_set):]
        else:
            test_set = samples[int(len(samples) * (training_set_percent + validating_set_percent)):]

        logger.info("Test set contains {} samples", len(test_set))
        test_X = Utils.prepare_X(arguments=test_set, word_embedding_dict=word_vector_map,
                                 word_embedding_length=embedding_size, max_seq_len=max_seq_len, frame_set=frames,
                                 filter_unknown_frames=filter_unknown_frames, using_topic=using_topic,
                                 using_premise=using_premise, using_conclusion=using_conclusion)
        if using_word_embedding_output:
            test_Y = Utils.compute_y_word_embedding(samples=test_set, word_vector_map=word_vector_map_output,
                                                    embedding_length=embedding_size_output,
                                                    max_seq_len=max_seq_len_output)
        elif isinstance(one_hot_output_clusters, Frames.GenericFrame) and frames is None:
            test_Y = Utils.compute_y_user_label_to_generic_frame_distribution(samples=test_set,
                                                                              word2vec=word_vector_map,
                                                                              frames=one_hot_output_clusters,
                                                                              enable_fuzzy_framing=enable_fuzzy_framing,
                                                                              enable_other_class=one_hot_output_clusters == Frames.media_frames_set)
        else:
            test_Y = Utils.compute_y_frame_distribution(samples=test_set,
                                                        frames=frames if frames is not None else clusters,
                                                        enable_fuzzy_framing=enable_fuzzy_framing,
                                                        ignore_unknown=filter_unknown_frames)

        model_save_path = None
        if iterations >= 1:
            model = keras.Sequential(
                name="{}{}".format("Easy-{}".format(NN_which_used), "-W2Vec" if using_word_embedding_output else ""))
            # model.add(keras_layers.Input(name="Input", shape=(max_seq_len, embedding_size)))
            model.add(
                keras_layers.Masking(name="Padding_recognizer", mask_value=0.0,
                                     input_shape=(max_seq_len, embedding_size)))
            if NN_which_used == "CNN":
                model.add(keras_layers.Conv1D(name="brain1", filters=128, kernel_size=5, padding="causal"))
                model.add(keras_layers.MaxPool1D(name="brain2"))
                model.add(keras_layers.Dropout(name="regularization", rate=0.25))
                if not using_word_embedding_output or max_seq_len_output < 1:
                    model.add(keras_layers.GlobalMaxPooling1D(name="brain3"))
            elif NN_which_used == "BiLSTM":
                model.add(
                    keras_layers.Bidirectional(keras_layers.LSTM(name="brain",
                                                                 units=max_seq_len_output * 64
                                                                 if using_word_embedding_output and max_seq_len_output >= 1
                                                                 else 128,
                                                                 use_bias=True, stateful=False,
                                                                 return_sequences=False, dropout=0.2)))
            else:
                model.add(
                    keras_layers.Bidirectional(keras_layers.GRU(name="brain",
                                                                units=max_seq_len_output * 64
                                                                if using_word_embedding_output and max_seq_len_output >= 1
                                                                else 128,
                                                                use_bias=True, stateful=False,
                                                                return_sequences=False, dropout=0.2)))
            if using_word_embedding_output and max_seq_len_output >= 1:
                if NN_which_used == "CNN":
                    assert max_seq_len_output < max_seq_len
                    model.add(keras_layers.Conv1D(name="brain3", filters=64,
                                                  kernel_size=int((max_seq_len / 2) - max_seq_len_output + 1),
                                                  padding="valid"))
                elif NN_which_used == "BiLSTM":
                    model.add(keras_layers.Reshape(target_shape=(max_seq_len_output, 128)))
                    model.add(keras_layers.LSTM(units=64, use_bias=True, stateful=False,
                                                return_sequences=True))
                else:
                    model.add(keras_layers.Reshape(target_shape=(max_seq_len_output, 128)))
                    model.add(keras_layers.GRU(units=64, use_bias=True, stateful=False,
                                               return_sequences=True))
                model.add(keras_layers.TimeDistributed(
                    keras_layers.Dense(name="Token_predictor", units=embedding_size_output, activation="linear")))
            else:
                # noinspection PyUnresolvedReferences
                model.add(
                    keras_layers.Dense(name="Token_predictor" if using_word_embedding_output else "Frame_predictor",
                                       units=embedding_size_output if using_word_embedding_output else
                                       (frames.get_prediction_vector_length(ignore_unknown=filter_unknown_frames)
                                        if frames is not None else
                                        (clusters.get_y_length() if clusters is not None else
                                         one_hot_output_clusters.get_prediction_vector_length(
                                             ignore_unknown=one_hot_output_clusters != Frames.media_frames_set))),
                                       activation="linear" if using_word_embedding_output else (
                                           "sigmoid" if enable_fuzzy_framing else "softmax")))

            if using_word_embedding_output:
                model.compile(optimizer="rmsprop",
                              loss="cosine_similarity",
                              metrics=[
                                  "CosineSimilarity",
                                  UserLabelPredictionMetric(word2vec_dict=word_vector_map_output,
                                                            train_user_labels=test_set if training_set is None
                                                            else training_set)
                              ])
            else:
                model.compile(optimizer="adam",
                              loss="categorical_crossentropy" if not enable_fuzzy_framing else "cosine_similarity",
                              metrics=["categorical_accuracy"] if not enable_fuzzy_framing else ["mse", "accuracy"])

            logger.info("Model created!")
            model.summary()

            logger.info("Train it now!")

            early_stopping_threshold = 60000 if execute_on_ssh_compute else 30000
            # noinspection PyUnboundLocalVariable
            model.fit(x=training_set_X, y=training_set_Y, shuffle=shuffle_samples,
                      validation_data=(val_set_X, val_set_Y),
                      batch_size=batch_size, validation_batch_size=batch_size * 2, verbose=1, epochs=epochs,
                      callbacks=[
                          keras.callbacks.EarlyStopping(patience=2 if len(samples) <= early_stopping_threshold else 1,
                                                        restore_best_weights=True)])

            logger.info("Model trained")
            if i == iterations:
                # noinspection PyUnresolvedReferences
                model_save_path =\
                    Utils.save_model(model=model,
                                     model_save_path=pathlib.Path("trained_model",
                                                                  "{}FrameTask".format(
                                                                      "" if frames is None else "Generic")
                                                                  if not using_word_embedding_output
                                                                  else "SpecificFrameTask-{}{}".format(
                                                                      embedding_size_output,
                                                                      "avg" if max_seq_len_output <= 0 else
                                                                      "x{}".format(max_seq_len_output)),
                                                                  NN_which_used).
                                     joinpath(
                                         "{}-{}-{}-{}{}-{}-{}-{}-{}".format(data_set.stem,
                                                                            max_seq_len,
                                                                            word_embeddings.stem,
                                                                            word_embeddings_output.stem
                                                                            if using_word_embedding_output else
                                                                            ((
                                                                                 one_hot_output_clusters.name_of_frame_set
                                                                                 if clusters is None else clusters)
                                                                             if frames is None else
                                                                             frames.name_of_frame_set),
                                                                            "" if filter_unknown_frames or using_word_embedding_output else "+1",
                                                                            "fuzzy" if enable_fuzzy_framing else "0010",
                                                                            "topic" if using_topic else "",
                                                                            "premise" if using_premise else "",
                                                                            "conclusion" if using_conclusion else "")),
                                     additional_metrics_to_plot=["cosine_similarity",
                                                                 "val_cosine_similarity",
                                                                 "val_user_label_w2v_accuracy"]
                                     if using_word_embedding_output else [])

        # Prediction test:
        test_result = model.evaluate(x=test_X, y=test_Y, batch_size=batch_size, verbose=1)
        logger.warning("Tested the model. {} loss with {}% metric", test_result[0],
                       (round(test_result[-1] * 100.0, 1)))
        losses.append(test_result[0])
        accuracies.append(test_result[1])
        if using_word_embedding_output:
            target_word_vectors = Utils.return_user_label_specific_word2vec_embedding(
                word2vec_dict=word_vector_map_output,
                train_user_labels=test_set if training_set is None else training_set,
                embedding_length=embedding_size_output
            )
        else:
            target_word_vectors = dict()
        for i in range(10):
            X = Utils.prepare_X(arguments=samples[i:i + 1], word_embedding_dict=word_vector_map,
                                word_embedding_length=embedding_size, max_seq_len=max_seq_len, frame_set=frames,
                                filter_unknown_frames=filter_unknown_frames, using_topic=using_topic,
                                using_premise=using_premise, using_conclusion=using_conclusion)
            argument = " -- ".join(
                Utils.argument_to_str(samples[i], using_topic=using_topic, using_premise=using_premise,
                                      using_conclusion=using_conclusion))
            if not execute_on_ssh_compute:
                # noinspection PyUnresolvedReferences
                logger.warning("Output frame vector of the argument \"{}\" (shape {}): {} (should be {})",
                               argument,
                               X.shape,
                               Utils.calculates_predicted_words_specific_frame(model(X, training=False),
                                                                               target_word2vec=target_word_vectors,
                                                                               embedding_size=embedding_size_output)
                               if using_word_embedding_output else model(X, training=False),
                               samples[i].get("frame", "n/a")
                               if using_word_embedding_output else
                               (frames.decode_frame_label(samples[i].get("genericFrame", frames.frame_names[-1]),
                                                          ignore_unknown=filter_unknown_frames)
                                if frames is not None else
                                (clusters.get_y(samples[i].get("frame", "__UNKNOWN__"))
                                 if clusters is not None else
                                 Utils.compute_y_user_label_to_generic_frame_distribution([samples[i]],
                                                                                          word2vec=word_vector_map,
                                                                                          frames=one_hot_output_clusters,
                                                                                          enable_fuzzy_framing=enable_fuzzy_framing))
                                )
                               )

    avg_loss = (sum(losses) * 1.0) / len(losses)
    avg_acc = (sum(accuracies) * 1.0) / len(accuracies)
    if model_save_path is not None:
        Utils.add_plot_description(additional_text="Loss: {} (-{}+{}) Acc: {}% (-{}+{})".format(
            round(avg_loss, 3),
            round(avg_loss - min(losses), 2),
            round(max(losses) - avg_loss, 2),
            round(avg_acc * 100.0, 2),
            round((avg_acc - min(accuracies)) * 100.0, 1),
            round((max(accuracies) - avg_acc) * 100.0, 1)
        ), model_save_path=model_save_path)
