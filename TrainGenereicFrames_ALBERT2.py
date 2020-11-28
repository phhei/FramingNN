"""
Modern approach via
- https://www.kaggle.com/stitch/albert-in-keras-tf2-using-huggingface-explained
- https://medium.com/analytics-vidhya/bert-in-keras-tensorflow-2-0-using-tfhub-huggingface-81c08c5f81d8
"""

import csv
import datetime
import os
import pathlib
import random
import sys
from typing import List, Dict

import loguru
import tensorflow.keras as keras
import tensorflow.keras.layers as keras_layers
import transformers

import Frames
import Utils
from Utils import compute_y_frame_distribution, argument_to_str
from _deprecated.TrainGenericFrames_ALBERT import single_albert_inputs_to_multiple_sample_input

# ####################################
# #### PARAMETERS ####################
# ####################################
max_seq_len = 48  # very important! Best situation is when no sentences is clipped
enable_fuzzy_framing = False
using_topic = False
using_premise = True
using_conclusion = True
split_parts_in_NN = False
filter_unknown_frames = False
frames = Frames.media_frames_set
model_tag = "albert-large-v2"  # lange model, see <https://huggingface.co/transformers/model_summary.html#autoencoding-models>

# Training - fit
batch_size = 8  # 4,8,16 (if you training for specific groups like stackoverflow then might be 32)
epochs = 2  # Epochs - range between 3,4

# Dataset
# dataset = pathlib.Path("Corpora", "Webis-argument-framing_out.csv")
data_set = pathlib.Path("Corpora").joinpath("MediaFramesCorpus").joinpath("converted").joinpath(
    "Media-immigrationsamesexsmoking-framing_gold_pure+exact_out.csv")
shuffle_samples = True
training_set_percent = 0.8
validating_set_percent = 0.1
test_set_percent = 0.1
under_sampling = False

######################################
# FUNCTIONS###########################
######################################

logger = loguru.logger
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="WARNING")
logger.info("Work with ALBERT model {}", model_tag)
logger.add(os.path.join(".", "logs", "{}_{}{}".format(model_tag,
                                                      str(int(datetime.datetime.timestamp(datetime.datetime.now()))),
                                                      ".log")), level="INFO", encoding="UTF-8")

language_model = model_tag
tokenizer = transformers.AlbertTokenizer.from_pretrained(pretrained_model_name_or_path=language_model,
                                                         cache_dir=".models",
                                                         force_download=False,
                                                         do_lower_case=True,
                                                         add_prefix_space=False
                                                         # only True with RoBERTa, in other cases set it to False
                                                         )


def prepare_arguments(argument_mappings: List[Dict]):
    logger.info("Prepare ALBERT-inputs with the Tokenizer: {}", tokenizer)
    input_ids_samples = []
    attention_mask = []
    token_type_ids_samples = []

    for argument_mapping in argument_mappings:
        if filter_unknown_frames and argument_mapping.get("genericFrame", "None") == "__UNKNOWN__":
            logger.debug("Filter \"{}\", because it's label is {}", ". ".join(
                argument_to_str(argument_mapping, using_topic=using_topic, using_premise=using_premise,
                                using_conclusion=using_conclusion)
            ), argument_mapping.get("genericFrame", "not encoded"))
        else:
            if split_parts_in_NN and sum([using_topic, using_premise, using_conclusion]) > 1:
                text_before = "{}: ".format(argument_mapping["topic"]) if using_topic else ""
                text_before += argument_mapping["premise"] if using_premise and using_topic else "None"
                text_main = argument_mapping["conclusion"] if using_conclusion else argument_mapping["premise"]
                if len(text_before) + len(text_main) == 0:
                    logger.warning("Empty mappings in {}", argument_mapping)
                prepared_input = tokenizer.encode_plus(text=text_before if len(text_before) > 0 else "None",
                                                       text_pair=text_main if len(text_main) > 0 else "None",
                                                       add_special_tokens=True,
                                                       max_length=max_seq_len,
                                                       pad_to_max_length=True,
                                                       return_token_type_ids=True,
                                                       return_attention_mask=True,
                                                       return_overflowing_tokens=True,
                                                       truncation=True)
            else:
                components = []
                if using_topic:
                    components.append(argument_mapping["topic"])
                if using_premise:
                    components.append(argument_mapping["premise"])
                if using_conclusion:
                    components.append(argument_mapping["conclusion"])
                text = ". ".join(components)
                if len(text) == 0:
                    logger.warning("Should encode an empty string (argument : {})! Set to \"None\"", argument_mapping)
                    text = "None"
                prepared_input = tokenizer.encode_plus(text=text,
                                                       add_special_tokens=True,
                                                       max_length=max_seq_len,
                                                       pad_to_max_length=True,
                                                       return_token_type_ids=True,
                                                       return_attention_mask=True,
                                                       return_overflowing_tokens=True,
                                                       truncation=True)

            input_ids_samples.append(prepared_input["input_ids"])
            attention_mask.append(prepared_input["attention_mask"])
            token_type_ids_samples.append(prepared_input["token_type_ids"])

            logger.debug("Processed an argument map: {}. Output is {} / {}. Truncated {} tokens.", argument_mapping,
                         prepared_input["input_ids"], prepared_input["token_type_ids"],
                         prepared_input.get("num_truncated_tokens", 0))
            if prepared_input.get("num_truncated_tokens", 0) > max_seq_len:
                logger.warning("Truncated {} tokens. Consider a increase of the max_token_length: {}",
                               prepared_input.get("num_truncated_tokens", 0), max_seq_len)

    logger.info("Processed {} samples: Collected {} token ids!", len(argument_mappings),
                sum(map(lambda id_sample: len(id_sample), input_ids_samples)))

    return input_ids_samples, attention_mask, token_type_ids_samples


def build_NN() -> keras.Model:
    token_inputs = keras_layers.Input((max_seq_len,), dtype="int32", name="input_word_ids")
    attention_ids = keras_layers.Input((max_seq_len,), dtype="int32", name="masks")
    types_inputs = keras_layers.Input((max_seq_len,), dtype="int32", name="input_types")

    transformer_model_layer = transformers.TFAlbertModel.from_pretrained(pretrained_model_name_or_path=model_tag,
                                                                         cache_dir=".models", force_download=False)
    transformer_model_layer.trainable = False

    # going with pooled output since seq_output results in ResourceExhausted Error even with GPU
    _, pooled_output = transformer_model_layer([token_inputs, attention_ids, types_inputs])
    X = keras_layers.Dropout(rate=0.2, name="Dropout_Regulator")(pooled_output)
    # dense_connector = keras_layers.Dense(units=frames.get_prediction_vector_length()*8, activation="tanh", name="hidden_dense")(X)
    if enable_fuzzy_framing:
        output_ = keras_layers.Dense(units=frames.get_prediction_vector_length(ignore_unknown=filter_unknown_frames),
                                     activation="sigmoid", name="output")(X)
    else:
        output_ = keras_layers.Dense(units=frames.get_prediction_vector_length(ignore_unknown=filter_unknown_frames),
                                     activation="softmax", name="output")(X)

    albert_model2 = keras.Model([token_inputs, attention_ids, types_inputs], output_)

    if enable_fuzzy_framing:
        albert_model2.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001), loss="cosine_similarity",
                              metrics=["KLDivergence", "categorical_accuracy" if filter_unknown_frames else "accuracy"])
    else:
        albert_model2.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001),
                              loss="categorical_crossentropy",
                              metrics=["categorical_accuracy"])

    albert_model2.summary()

    logger.info("Created neural net now: {}", albert_model2)

    return albert_model2


####################################

if __name__ == "__main__":
    logger.info("Read data set at {}", os.path.abspath(data_set))
    data = Utils.load_csv(data_set=data_set.absolute(), frames=frames, filter_unknown_frames=filter_unknown_frames,
                          shuffle_samples=shuffle_samples, under_sampling=under_sampling, limit_data=2500)

    if 0 < training_set_percent <= 1:
        training_set = data[:round(len(data) * training_set_percent)]
        training_set_X, training_set_X_masks, training_set_X_types = prepare_arguments(training_set)
        training_set_X = single_albert_inputs_to_multiple_sample_input(training_set_X, max_seq_len)
        training_set_X_masks = single_albert_inputs_to_multiple_sample_input(training_set_X_masks, max_seq_len)
        training_set_X_types = single_albert_inputs_to_multiple_sample_input(training_set_X_types, max_seq_len)
        training_set_Y = compute_y_frame_distribution(samples=training_set, frames=frames,
                                                      ignore_unknown=filter_unknown_frames,
                                                      enable_fuzzy_framing=enable_fuzzy_framing)
    else:
        training_set = []
        training_set_X = []
        training_set_X_types = []
        training_set_X_masks = []
        training_set_Y = []
        logger.critical("Chosen no training set. The chosen percent is {}", training_set_percent * 100)
    if 0 < validating_set_percent <= 1:
        validating_set = data[len(training_set):len(training_set) + round(len(data) * validating_set_percent)]
        validating_set_X, validating_set_X_masks, validating_set_X_types = prepare_arguments(validating_set)
        validating_set_X = single_albert_inputs_to_multiple_sample_input(validating_set_X, max_seq_len)
        validating_set_X_masks = single_albert_inputs_to_multiple_sample_input(validating_set_X_masks, max_seq_len)
        validating_set_X_types = single_albert_inputs_to_multiple_sample_input(validating_set_X_types, max_seq_len)
        validating_set_Y = compute_y_frame_distribution(samples=validating_set, frames=frames,
                                                        ignore_unknown=filter_unknown_frames,
                                                        enable_fuzzy_framing=enable_fuzzy_framing)
    else:
        validating_set = []
        validating_set_X = []
        validating_set_X_masks = []
        validating_set_X_types = []
        validating_set_Y = []
        logger.warning("Chosen no validating set. The chosen percent is {}", validating_set_percent * 100)
    if 0 < test_set_percent <= 1:
        test_set = data[len(training_set) + len(validating_set):]
        test_set_X, test_set_masks, test_set_X_types = prepare_arguments(test_set)
        test_set_X = single_albert_inputs_to_multiple_sample_input(test_set_X, max_seq_len)
        test_set_X_masks = single_albert_inputs_to_multiple_sample_input(test_set_masks, max_seq_len)
        test_set_X_types = single_albert_inputs_to_multiple_sample_input(test_set_X_types, max_seq_len)
        test_set_Y = compute_y_frame_distribution(samples=test_set, frames=frames, ignore_unknown=filter_unknown_frames,
                                                  enable_fuzzy_framing=enable_fuzzy_framing)
    else:
        test_set = []
        test_set_X = []
        test_set_X_masks = []
        test_set_X_types = []
        test_set_Y = []
        logger.warning("Chosen no training set. The chosen percent is {}", test_set * 100)

    logger.info("Train model now...")

    model = build_NN()

    model.fit(x=[training_set_X, training_set_X_masks, training_set_X_types],
              y=training_set_Y,
              batch_size=batch_size, epochs=epochs, shuffle=shuffle_samples,
              validation_data=([validating_set_X, validating_set_X_masks, validating_set_X_types], validating_set_Y))

    model_save_path = pathlib.Path("trained_model").joinpath("_".join(
        [model_tag, str(max_seq_len), str(enable_fuzzy_framing) + "ff", "pre" + str(using_premise),
         "con" + str(using_conclusion), "top" + str(using_topic), model_tag + str(split_parts_in_NN)]))
    if os.path.exists(model_save_path):
        logger.warning("The model file vor saving \"{}\" is already blocked - delete the previous model",
                       os.path.basename(model_save_path))
    Utils.save_model(model=model, model_save_path=model_save_path)
    # Prediction test:
    test_result = model.evaluate(x=[test_set_X, test_set_X_masks, test_set_X_types],
                                 y=test_set_Y, batch_size=batch_size)
    logger.warning("Tested the model. {} loss with {}% metric", test_result[0],
                   (round(test_result[-1] * 1000.0) / 10.0))
    for show_sample in data[0:10]:
        argument = " -- ".join(argument_to_str(show_sample, using_topic=using_topic, using_premise=using_premise,
                                               using_conclusion=using_conclusion))
        X1, X2, X3 = prepare_arguments([show_sample])
        X1 = single_albert_inputs_to_multiple_sample_input(X1, max_seq_len)
        X2 = single_albert_inputs_to_multiple_sample_input(X2, max_seq_len)
        X3 = single_albert_inputs_to_multiple_sample_input(X3, max_seq_len)
        logger.warning("Output frame vector of the argument \"{}\" (code {}): {} (should be {})", argument, X1,
                       model([X1, X2, X3], training=False),
                       frames.decode_frame_label(show_sample.get("genericFrame", frames.frame_names[-1]),
                                                 ignore_unknown=filter_unknown_frames))
