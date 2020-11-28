"""
Beware of the Params-section right below the imports!

- general in- and output of the NN
  - Inputs / Shapes/ Preprocessing
    - max_seq_len
    - what to use from the plain input
      - using_topic
      - using_premise
      - using_conclusion
      - word_embeddings_input + embedding_size_input
      - word_embeddings_output: with having a number here, you activate the mode of predicting word embeddings
      (requires embedding_size_output)
      - frames: the Frames.GenericFrame-set for the Media.Frames-task-prediction
      - one_hot_output_clusters:
        - with having a number here, a K-MeansClusterer will cluster all user labels for the Webis-task ==> one-hot-
        encoding
        - with having a Frames.GenericFrame-set here, the word-embedding next to a generic frame will be predicted
        (without fuzzy framing)
- Settings for the NN (architecture)
  - NN_which_used
  - give_webis_extra_layers (only used with RNNs and one-hot-encodings): added 2 layers on top to the webis branch:
    one heavy Dropout (seems to be necessary)
    one extra Dense layer with bias neuron
  - soft_parameter_sharing_lambda: there is one layer in common between the two tasks. Regulated the Loss function,
    hence the impact of the parameter sharing
  - batch_size
  - epochs
  - iterations
- regarding the corpora
  Besides the data sets
  - under_sampling_dataset: very crucial param! If True, we will have only few data, with False we have lots of data
  - under_sampling_classes: reduces IN a dataset to that class which have the lowest coverage
  - shuffle_samples: to shuffle the samples
  - and the parts of training, validation and test in %/100 (so for the half write 0.5)
"""

from typing import Union

import Losses
from Metric_UserLabel import UserLabelPredictionMetric

# We executed the code on strong CPU clusters without an GPU (ssh compute). Because of this extraordinary
# executing environment, we introduce this flag. To reproduce the results in the paper, enable this flag.
execute_on_ssh_compute = False

import pathlib
import sys

import loguru
import nltk
import tensorflow.keras as keras
import tensorflow.keras.layers as keras_layers

logger = loguru.logger
logger.remove()
logger.add(sink=sys.stdout, level="INFO", colorize=True, backtrace=True, catch=True)

import Frames
import Utils

# ######################################################################################################################

max_seq_len = 256  # very important! Best situation is when no sample is clipped (default = 500)
using_topic = False
using_premise = True
using_conclusion = True
filter_unknown_frames = False  # should be FALSE!
word_embeddings_path_base = pathlib.Path.home().joinpath("remote", "glove") \
    if execute_on_ssh_compute else pathlib.Path.home().joinpath("Documents", "Einsortiertes",
                                                                "Nach Lebensabschnitten einsortiert",
                                                                "Promotionszeit (2019-2023)", "Promotion",
                                                                "Programming", "_wordEmbeddings", "glove")
word_embeddings_input = word_embeddings_path_base.joinpath("glove.840B.300d.txt")
embedding_size_input = 300
word_embeddings_output = None  # if None, the one-hot-encoded embedded frame is predicted
max_seq_len_output = 2  # a negative value leads to averaging the different word vectors of the frame tokens.
embedding_size_output = 50
frames = Frames.media_frames_set
one_hot_output_clusters: Union[int, Frames.GenericFrame]
one_hot_output_clusters = Frames.media_frames_set

# Training - fit
trained_model = None
# trained_model = pathlib.Path(
#    "trained_model",
#    "MultiTask",
#    "BiGRU",
#    "_Media-immigrationsamesexsmoking-framing_dirty_all+sentexact_out-92-glove.6B.300d-MediaFramesSet (most frequent)-fuzzy--premise-conclusion"
# )
NN_which_used = "BiGRU"  # "BiGRU", "BiLSTM" or "CNN" - Naderi & Hirst showed better results with GRUs
assert NN_which_used in ["BiGRU", "BiLSTM", "CNN"]
give_webis_extra_layers = False
soft_parameter_sharing_lambda = 0.2  # if >= 1, hard parameter sharing is enabled
batch_size = 32
epochs = 12
iterations = 12

# Dataset
data_set_argument = pathlib.Path("Corpora", "Webis-argument-framing_out.csv")
# data_set_argument = pathlib.Path("Corpora", "Webis-argument-framing-mediaFramesTopics_out.csv")
data_set_media_frames = pathlib.Path("Corpora").joinpath("MediaFramesCorpus").joinpath("converted").joinpath(
    "Media-immigrationsamesexsmoking-framing_gold_pure+exact_out.csv")
under_sampling_dataset = not execute_on_ssh_compute
shuffle_samples = True
training_set_percent = 0.8
validating_set_percent = 0.1
test_set_percent = 0.1
under_sampling_classes = False

#######################################################################################################################

nltk.download("punkt")

#######################################################################################################################

if __name__ == "__main__":
    samples_webis = Utils.load_csv(data_set=data_set_argument, frames=frames,
                                   filter_unknown_frames=filter_unknown_frames,
                                   shuffle_samples=shuffle_samples and word_embeddings_output is not None,
                                   under_sampling=under_sampling_classes)
    samples_media = Utils.load_csv(data_set=data_set_media_frames, frames=frames,
                                   filter_unknown_frames=filter_unknown_frames,
                                   shuffle_samples=shuffle_samples, under_sampling=under_sampling_classes)

    word_vector_map_input = Utils.load_word_embeddings(glove_file=word_embeddings_input,
                                                       embedding_size=embedding_size_input)

    model = Utils.load_pre_trained_model(path=trained_model)
    if model is not None:
        iterations = -1

    acc_media = []
    acc_webis = []

    clusters = None
    model_save_path = None
    training_set = None
    test_set = None
    for sample_index in range(1, max(2, iterations + 1)):
        logger.info("Iteration {}", sample_index)
        if word_embeddings_output is None and isinstance(one_hot_output_clusters, int):
            clusters = \
                Utils.UserLabelCluster(user_labels=[sample.get("frame", "n/a")
                                                    for sample in
                                                    samples_webis[
                                                    : int(len(samples_webis) * training_set_percent)]],
                                       word2vec_dict=word_vector_map_input,
                                       cluster_k=one_hot_output_clusters,
                                       word2vec_embedding_size=300,
                                       semantic_clustering=True,
                                       iteration = sample_index
                                       )
            logger.info("Found {} distinct frames in test set. That are {} clusters",
                        len(clusters.classes), clusters.get_y_length())
        elif word_embeddings_output is not None:
            # noinspection PyTypeChecker
            word_vector_map_output = Utils.load_word_embeddings(glove_file=word_embeddings_output,
                                                                embedding_size=embedding_size_output)
            logger.info("However, we loaded {} word embeddings instead", len(word_vector_map_output))
        training_set = None
        if iterations >= 1:
            training_set = Utils.to_same_sample_amount([samples_webis[:int(len(samples_webis) * training_set_percent)],
                                                        samples_media[:int(len(samples_media) * training_set_percent)]],
                                                       under_sampling=under_sampling_dataset)
            training_set_webis_task = training_set[0]
            training_set_media_task = training_set[1]
            logger.info("Training set contains {} samples", len(training_set_webis_task))
            training_set_X_webis_task = Utils.prepare_X(arguments=training_set_webis_task,
                                                        word_embedding_dict=word_vector_map_input,
                                                        word_embedding_length=embedding_size_input,
                                                        max_seq_len=max_seq_len,
                                                        filter_unknown_frames=filter_unknown_frames,
                                                        frame_set=frames,
                                                        using_topic=using_topic, using_premise=using_premise,
                                                        using_conclusion=using_conclusion)
            training_set_X_media_task = Utils.prepare_X(arguments=training_set_media_task,
                                                        word_embedding_dict=word_vector_map_input,
                                                        word_embedding_length=embedding_size_input,
                                                        max_seq_len=max_seq_len,
                                                        filter_unknown_frames=filter_unknown_frames,
                                                        frame_set=frames,
                                                        using_topic=using_topic, using_premise=using_premise,
                                                        using_conclusion=using_conclusion)
            # noinspection PyUnboundLocalVariable
            training_set_Y_webis_task = Utils.compute_y_word_embedding(samples=training_set_webis_task,
                                                                       word_vector_map=word_vector_map_output,
                                                                       embedding_length=embedding_size_output,
                                                                       max_seq_len=max_seq_len_output,
                                                                       filter_stop_words=True) \
                if word_embeddings_output is not None else \
                (Utils.compute_y_frame_distribution(samples=training_set_webis_task,
                                                    frames=clusters,
                                                    ignore_unknown=filter_unknown_frames,
                                                    enable_fuzzy_framing=False)
                 if clusters is not None else
                 Utils.compute_y_user_label_to_generic_frame_distribution(samples=training_set_webis_task,
                                                                          word2vec=word_vector_map_input,
                                                                          frames=one_hot_output_clusters,
                                                                          enable_other_class=one_hot_output_clusters == Frames.media_frames_set))
            training_set_Y_media_task = Utils.compute_y_frame_distribution(samples=training_set_media_task,
                                                                           frames=frames,
                                                                           enable_fuzzy_framing=False,
                                                                           ignore_unknown=filter_unknown_frames)

            validation_set = Utils.to_same_sample_amount([samples_webis[int(len(samples_webis) * training_set_percent):
                                                                        int(len(samples_webis) *
                                                                            (training_set_percent +
                                                                             validating_set_percent))],
                                                          samples_media[int(len(samples_media) * training_set_percent):
                                                                        int(len(samples_media) *
                                                                            (training_set_percent +
                                                                             validating_set_percent))]],
                                                         under_sampling=under_sampling_dataset)
            validation_set_webis_task = validation_set[0]
            validation_set_media_task = validation_set[1]
            logger.info("Validation set contains {} samples", len(validation_set_webis_task))
            validation_set_X_webis_task = Utils.prepare_X(arguments=validation_set_webis_task,
                                                          word_embedding_dict=word_vector_map_input,
                                                          word_embedding_length=embedding_size_input,
                                                          max_seq_len=max_seq_len,
                                                          filter_unknown_frames=filter_unknown_frames, frame_set=frames,
                                                          using_topic=using_topic, using_premise=using_premise,
                                                          using_conclusion=using_conclusion)
            validation_set_X_media_task = Utils.prepare_X(arguments=validation_set_media_task,
                                                          word_embedding_dict=word_vector_map_input,
                                                          word_embedding_length=embedding_size_input,
                                                          max_seq_len=max_seq_len,
                                                          filter_unknown_frames=filter_unknown_frames, frame_set=frames,
                                                          using_topic=using_topic, using_premise=using_premise,
                                                          using_conclusion=using_conclusion)
            validation_set_Y_webis_task = Utils.compute_y_word_embedding(samples=validation_set_webis_task,
                                                                         word_vector_map=word_vector_map_output,
                                                                         embedding_length=embedding_size_output,
                                                                         max_seq_len=max_seq_len_output,
                                                                         filter_stop_words=True) \
                if word_embeddings_output is not None else \
                (Utils.compute_y_frame_distribution(samples=validation_set_webis_task,
                                                    frames=clusters,
                                                    ignore_unknown=filter_unknown_frames,
                                                    enable_fuzzy_framing=False)
                 if clusters is not None else
                 Utils.compute_y_user_label_to_generic_frame_distribution(samples=validation_set_webis_task,
                                                                          word2vec=word_vector_map_input,
                                                                          frames=one_hot_output_clusters,
                                                                          enable_other_class=one_hot_output_clusters == Frames.media_frames_set))
            validation_set_Y_media_task = Utils.compute_y_frame_distribution(samples=validation_set_media_task,
                                                                             frames=frames,
                                                                             enable_fuzzy_framing=False,
                                                                             ignore_unknown=filter_unknown_frames)

        test_set = Utils.to_same_sample_amount(
            [samples_webis[int(len(samples_webis) * (training_set_percent + validating_set_percent)):],
             samples_media[int(len(samples_media) * (training_set_percent + validating_set_percent)):]],
            under_sampling=under_sampling_dataset)
        test_set_webis_task = test_set[0]
        test_set_media_task = test_set[1]
        logger.info("Test set contains {} samples", len(test_set_webis_task))
        test_set_X_webis_task = Utils.prepare_X(arguments=test_set_webis_task,
                                                word_embedding_dict=word_vector_map_input,
                                                word_embedding_length=embedding_size_input, max_seq_len=max_seq_len,
                                                filter_unknown_frames=filter_unknown_frames, frame_set=frames,
                                                using_topic=using_topic, using_premise=using_premise,
                                                using_conclusion=using_conclusion)
        test_set_X_media_task = Utils.prepare_X(arguments=test_set_media_task,
                                                word_embedding_dict=word_vector_map_input,
                                                word_embedding_length=embedding_size_input, max_seq_len=max_seq_len,
                                                filter_unknown_frames=filter_unknown_frames, frame_set=frames,
                                                using_topic=using_topic, using_premise=using_premise,
                                                using_conclusion=using_conclusion)
        test_set_Y_webis_task = Utils.compute_y_word_embedding(samples=test_set_webis_task,
                                                               word_vector_map=word_vector_map_output,
                                                               embedding_length=embedding_size_output,
                                                               max_seq_len=max_seq_len_output,
                                                               filter_stop_words=True) \
            if word_embeddings_output is not None else \
            (Utils.compute_y_frame_distribution(samples=test_set_webis_task,
                                                frames=clusters,
                                                ignore_unknown=filter_unknown_frames,
                                                enable_fuzzy_framing=False)
             if clusters is not None else
             Utils.compute_y_user_label_to_generic_frame_distribution(samples=test_set_webis_task,
                                                                      word2vec=word_vector_map_input,
                                                                      frames=one_hot_output_clusters,
                                                                      enable_other_class=one_hot_output_clusters == Frames.media_frames_set))
        test_set_Y_media_task = Utils.compute_y_frame_distribution(samples=test_set_media_task, frames=frames,
                                                                   enable_fuzzy_framing=False,
                                                                   ignore_unknown=filter_unknown_frames)

        model_save_path = None
        if iterations >= 1:
            input_webis = keras_layers.Input(shape=(max_seq_len, embedding_size_input), name="input_webis")
            input_webis_masked = keras_layers.Masking(name="input_webis_masked", mask_value=0.0)(input_webis)
            input_media = keras_layers.Input(shape=(max_seq_len, embedding_size_input), name="input_media")
            input_media_masked = keras_layers.Masking(name="input_media_masked", mask_value=0.0)(input_media)
            brains = []
            for _ in range(1 if soft_parameter_sharing_lambda >= 1 else 2):
                if NN_which_used == "CNN":
                    brain = keras.Sequential()
                    brain.add(keras_layers.Conv1D(name="shared_brain", filters=128, kernel_size=5, padding="causal"))
                    brain.add(keras_layers.MaxPool1D(name="shared_brain_feature_selector"))
                    brain.add(keras_layers.Dropout(name="shared_regularization", rate=0.2))
                elif NN_which_used == "BiLSTM":
                    brain = keras_layers.Bidirectional(
                        keras_layers.LSTM(name="shared_brain",
                                          units=max_seq_len_output * 64
                                          if word_embeddings_output is not None and max_seq_len_output >= 1 else 128,
                                          use_bias=True,
                                          stateful=False,
                                          return_sequences=False,
                                          dropout=0.25))
                else:
                    brain = keras_layers.Bidirectional(
                        keras_layers.GRU(name="shared_brain",
                                         units=max_seq_len_output * 64
                                         if word_embeddings_output is not None and max_seq_len_output >= 1 else 128,
                                         use_bias=True,
                                         stateful=False,
                                         return_sequences=False,
                                         dropout=0.25))
                brains.append(brain)

            hidden_layer_webis = brains[0](input_webis_masked)
            hidden_layer_media = brains[-1](input_media_masked)

            final_units = clusters.get_y_length() \
                if word_embeddings_output is None and isinstance(one_hot_output_clusters, int) \
                else (one_hot_output_clusters.get_prediction_vector_length(
                ignore_unknown=one_hot_output_clusters != Frames.media_frames_set)
                      if word_embeddings_output is None else embedding_size_output)
            if max_seq_len_output <= 0 or word_embeddings_output is None:
                if NN_which_used == "CNN":
                    preprocessor_webis = keras_layers.GlobalMaxPooling1D(name="dim_reducer_webis")(hidden_layer_webis)
                elif give_webis_extra_layers or word_embeddings_output is not None:
                    preprocessor_webis_dense = keras_layers.Dense(name="after_process_webis_relu",
                                                                  units=final_units,
                                                                  use_bias=give_webis_extra_layers,
                                                                  activation="relu")(hidden_layer_webis)
                    preprocessor_webis = keras_layers.Dropout(name="after_process_webis_regularization",
                                                              rate=0.1 if under_sampling_dataset else 0.66)(
                        preprocessor_webis_dense)
                else:
                    preprocessor_webis = hidden_layer_webis
                predictor_webis = keras_layers.Dense(name="embedding_webis",
                                                     units=final_units,
                                                     activation="softmax" if word_embeddings_output is None else "linear",
                                                     use_bias=True)(preprocessor_webis)
            else:
                # token-wised embedding output
                if NN_which_used == "CNN":
                    hidden_task_specific_webis = keras_layers.Conv1D(name="webis_Decoder-CNN",
                                                                     filters=64,
                                                                     kernel_size=int((max_seq_len / 2) -
                                                                                     max_seq_len_output + 1),
                                                                     padding="valid")(hidden_layer_webis)
                else:
                    predictor_embedding_webis_preprocessing = keras_layers.Reshape(
                        target_shape=(max_seq_len_output, 128))(
                        hidden_layer_webis)
                    if NN_which_used == "BiLSTM":
                        hidden_task_specific_webis = keras_layers.LSTM(name="webis_Decoder-LSTM",
                                                                       units=64,
                                                                       use_bias=True,
                                                                       stateful=False,
                                                                       dropout=0.5 if under_sampling_dataset and
                                                                                      not give_webis_extra_layers else 0.66,
                                                                       return_sequences=True) \
                            (predictor_embedding_webis_preprocessing)
                    else:
                        hidden_task_specific_webis = keras_layers.GRU(name="webis_Decoder-GRU",
                                                                      units=64,
                                                                      use_bias=True,
                                                                      stateful=False,
                                                                      dropout=0.5 if under_sampling_dataset and
                                                                                     not give_webis_extra_layers else 0.66,
                                                                      return_sequences=True) \
                            (predictor_embedding_webis_preprocessing)
                predictor_webis = keras_layers.TimeDistributed(name="embedding_webis",
                                                               layer=keras_layers.Dense(
                                                                   units=embedding_size_output,
                                                                   activation="linear")
                                                               )(hidden_task_specific_webis)
            preprocessor_media = None
            if NN_which_used == "CNN":
                preprocessor = keras_layers.GlobalMaxPooling1D(name="dim_reducer_media")(hidden_layer_media)
            predictor_media = keras_layers.Dense(name="generic_frame_media",
                                                 units=frames.get_prediction_vector_length(
                                                     ignore_unknown=filter_unknown_frames),
                                                 activation="softmax")(
                hidden_layer_media if preprocessor_media is None else preprocessor_media)
            model = keras.Model(name="MTL-{}{}".format(NN_which_used,
                                                       "-WebisDenseDropout" if give_webis_extra_layers else ""),
                                inputs=(input_webis, input_media), outputs=(predictor_webis, predictor_media))

            loss_for_webis = keras.losses.CosineSimilarity() if word_embeddings_output is not None else \
                keras.losses.CategoricalCrossentropy()
            scalar_value = 1.0 if under_sampling_dataset else len(samples_webis) / (len(samples_media) + 1.0)
            # noinspection PyUnboundLocalVariable
            model.compile(optimizer="adam",
                          loss={
                              "embedding_webis": Losses.SoftParameterSharingLoss(
                                  base_loss=loss_for_webis,
                                  shared_layers=brains,
                                  impact_value=soft_parameter_sharing_lambda,
                                  scalar_value=scalar_value
                              ) if soft_parameter_sharing_lambda < 1 else
                              Losses.ScalingLoss(base_loss=loss_for_webis, scale_factor=scalar_value),
                              "generic_frame_media": Losses.SoftParameterSharingLoss(
                                  base_loss=keras.losses.CategoricalCrossentropy(),
                                  shared_layers=brains,
                                  impact_value=soft_parameter_sharing_lambda
                              ) if soft_parameter_sharing_lambda < 1 else keras.losses.CategoricalCrossentropy()},
                          metrics={"embedding_webis":
                                       ["CosineSimilarity",
                                        UserLabelPredictionMetric(word2vec_dict=word_vector_map_output,
                                                                  train_user_labels=test_set_webis_task
                                                                  if training_set_webis_task is None else
                                                                  training_set_webis_task)
                                        ] if word_embeddings_output is not None else ["categorical_accuracy"],
                                   "generic_frame_media": ["categorical_accuracy"]})

            logger.info("Model created!")
            model.summary()

            logger.info("Train it now!")

            early_stopping_threshold = 60000 if execute_on_ssh_compute else 30000
            # noinspection PyUnboundLocalVariable
            model.fit(x={"input_webis": training_set_X_webis_task, "input_media": training_set_X_media_task},
                      y={"embedding_webis": training_set_Y_webis_task,
                         "generic_frame_media": training_set_Y_media_task},
                      shuffle=shuffle_samples,
                      validation_data=(
                          {"input_webis": validation_set_X_webis_task, "input_media": validation_set_X_media_task},
                          {"embedding_webis": validation_set_Y_webis_task,
                           "generic_frame_media": validation_set_Y_media_task}),
                      batch_size=batch_size,
                      validation_batch_size=batch_size * 2,
                      verbose=1,
                      epochs=epochs,
                      callbacks=[keras.callbacks.EarlyStopping(
                          patience=2
                          if len(training_set_X_webis_task) <= early_stopping_threshold or execute_on_ssh_compute
                          else 1,
                          restore_best_weights=True)])

            logger.info("Model trained")
            if sample_index == iterations:
                # noinspection PyUnresolvedReferences
                model_save_path = Utils.save_model(model=model,
                                                   additional_metrics_to_plot=["val_embedding_webis_cosine_similarity",
                                                                               "val_embedding_webis_user_label_w2v_accuracy",
                                                                               "val_generic_frame_media_categorical_accuracy",
                                                                               "generic_frame_media_categorical_accuracy"]
                                                   if word_embeddings_output is not None else
                                                   ["val_embedding_webis_categorical_accuracy",
                                                    "embedding_webis_categorical_accuracy",
                                                    "val_generic_frame_media_categorical_accuracy",
                                                    "generic_frame_media_categorical_accuracy"],
                                                   model_save_path=pathlib.Path("trained_model", "MultiTask",
                                                                                NN_which_used).
                                                   joinpath("{}-{}-{}.{}{}-{}{}.{}-{}-{}".format(
                                                       "{}{}{}".format(data_set_media_frames.stem,
                                                                       soft_parameter_sharing_lambda
                                                                       if soft_parameter_sharing_lambda < 1 else "",
                                                                       "" if data_set_argument.stem == "Webis-argument-framing_out" else "ts"),
                                                       max_seq_len,
                                                       word_embeddings_input.stem,
                                                       clusters
                                                       if word_embeddings_output is None and
                                                          isinstance(one_hot_output_clusters, int)
                                                       else (one_hot_output_clusters.name_of_frame_set
                                                             if word_embeddings_output is None else
                                                             word_embeddings_output.stem),
                                                       "x{}".format(
                                                           max_seq_len_output) if max_seq_len_output >= 1 else "",
                                                       frames.name_of_frame_set,
                                                       "" if filter_unknown_frames else "+1",
                                                       "topic" if using_topic else "",
                                                       "prem" if using_premise else "",
                                                       "conc" if using_conclusion else "")))

        # Prediction test:
        test_result = model.evaluate(
            x={"input_webis": test_set_X_webis_task, "input_media": test_set_X_media_task},
            y={"embedding_webis": test_set_Y_webis_task, "generic_frame_media": test_set_Y_media_task},
            batch_size=batch_size,
            verbose=1)
        logger.warning("Tested the model. {} loss with {}% metric", test_result[0],
                       (round(test_result[-1] * 1000.0) / 10.0))
        acc_webis.append(test_result[-2])
        acc_media.append(test_result[-1])
        if word_embeddings_output is not None:
            target_word_vectors = Utils.return_user_label_specific_word2vec_embedding(
                word2vec_dict=word_vector_map_output,
                train_user_labels=samples_webis if training_set is None else training_set_webis_task,
                embedding_length=embedding_size_output)
            for sample_index in range(10):
                X_media = Utils.prepare_X(arguments=test_set_media_task[sample_index:sample_index + 1],
                                          word_embedding_dict=word_vector_map_input,
                                          word_embedding_length=embedding_size_input, max_seq_len=max_seq_len,
                                          frame_set=frames,
                                          filter_unknown_frames=filter_unknown_frames, using_topic=using_topic,
                                          using_premise=using_premise, using_conclusion=using_conclusion)
                argument_media = " -- ".join(
                    Utils.argument_to_str(test_set_media_task[sample_index], using_topic=using_topic,
                                          using_premise=using_premise, using_conclusion=using_conclusion))
                X_webis = Utils.prepare_X(arguments=test_set_webis_task[sample_index:sample_index + 1],
                                          word_embedding_dict=word_vector_map_input,
                                          word_embedding_length=embedding_size_input, max_seq_len=max_seq_len,
                                          frame_set=frames,
                                          filter_unknown_frames=filter_unknown_frames, using_topic=using_topic,
                                          using_premise=using_premise, using_conclusion=using_conclusion)
                argument_webis = " -- ".join(
                    Utils.argument_to_str(test_set_media_task[sample_index], using_topic=using_topic, using_premise=using_premise,
                                          using_conclusion=using_conclusion))
                output = model({"input_webis": X_webis, "input_media": X_media}, training=False)
                output_webis = output[0]
                output_media = output[1]
                text = "Output frame vector of the argument \"{}\" (shape {}): {} (should be {})".format(
                    argument_media,
                    X_media.shape,
                    output_media,
                    frames.decode_frame_label(
                        frame_name_distribution=test_set_webis_task[sample_index].get("genericFrame", frames.frame_names[-1]),
                        ignore_unknown=filter_unknown_frames)
                )
                if execute_on_ssh_compute:
                    logger.trace(text)
                else:
                    logger.info(text)
                text = "Output user label vector of the argument \"{}\" (shape {}): {} (should be {})".format(
                    argument_webis,
                    X_webis.shape,
                    Utils.calculates_predicted_words_specific_frame(output_webis, target_word2vec=target_word_vectors,
                                                                    embedding_size=embedding_size_output),
                    test_set_webis_task[sample_index].get("frame", frames.frame_names[-1])
                )
                if execute_on_ssh_compute:
                    logger.debug(text)
                else:
                    logger.warning(text)

    avg_acc_media = (sum(acc_media) * 1.0) / len(acc_media)
    avg_acc_webis = (sum(acc_webis) * 1.0) / len(acc_webis)
    if model_save_path is not None:
        Utils.add_plot_description(additional_text="Media: {}% (-{}+{}) | Webis: {}% (-{}+{})".format(
            round(avg_acc_media * 100.0, 2),
            round((avg_acc_media - min(acc_media)) * 100.0, 1),
            round((max(acc_media) - avg_acc_media) * 100.0, 1),
            round(avg_acc_webis * 100.0, 2),
            round((avg_acc_webis - min(acc_webis)) * 100.0, 1),
            round((max(acc_webis) - avg_acc_webis) * 100.0, 1)
        ), model_save_path=model_save_path)
