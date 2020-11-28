"""
Not all metrics can be expressed via stateless callables, because metrics are evaluated for each batch during training
and evaluation, but in some cases the average of the per-batch values is not what you are interested in.

a metric that makes evaluates user label predictions with the help of word vectors (glove)
"""

import nltk
import numpy
import tensorflow
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import variables

from Utils import calculates_predicted_words_specific_frame, return_user_label_specific_word2vec_embedding

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords

setStopWords = set(stopwords.words("english"))


# noinspection SpellCheckingInspection
class UserLabelPredictionMetric(Metric):

    def __init__(self, word2vec_dict: dict, train_user_labels: [dict], name="user_label_w2v_accuracy", **kwargs):
        super(UserLabelPredictionMetric, self).__init__(name=name, **kwargs)
        self.metric_value_sum = self.add_weight(name="w2v_accuracy", aggregation=variables.VariableAggregation.SUM,
                                                initializer="zeros", dtype="float32")
        self.metric_value_count = self.add_weight(name="w2v_count",  # shape=(1,),
                                                  aggregation=variables.VariableAggregation.SUM, initializer="zeros",
                                                  dtype="int32")

        self.target_word_vectors = return_user_label_specific_word2vec_embedding(word2vec_dict=word2vec_dict,
                                                                                 train_user_labels=train_user_labels)

    def update_state(self, y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor, sample_weight=None):
        # print("Update stat...")
        y_true = tensorflow.cast(y_true, tensorflow.float32)
        y_pred = tensorflow.cast(y_pred, tensorflow.float32)

        try:
            y_prediction_numpy = tensorflow.make_ndarray(y_pred)
            y_true_numpy = tensorflow.make_ndarray(y_true)
            values = numpy.zeros(shape=(y_true.shape[0],), dtype=tensorflow.float32)
            for i in range(y_pred.shape[0]):
                prediction = [p[1] for p in
                              calculates_predicted_words_specific_frame(word_vectors_prediction=y_prediction_numpy[i],
                                                                        target_word2vec=self.target_word_vectors)]
                equals = [numpy.allclose(prediction[p], y_true_numpy[i, p]) for p in range(len(prediction))]
                values[i] = float(sum(equals)) / len(equals)
        except AttributeError:
            # happens by symbolic tensors. Symbolic tensor are tensors were a shape is not absolutely defined (None),
            # so speak: a batch size, which differs (in common training scenario this is the case)
            # https://pgaleone.eu/tensorflow/2018/07/28/understanding-tensorflow-tensors-shape-static-dynamic/ :
            #   with tensorflow.shape(x), we fetch the dynamic dimension, hence, the actual batch size =)

            def process_batch(v, step):
                shape_tensors = tensorflow.shape(y_true)
                # tensorflow.print(shape_tensors)
                # sys.stderr.write("Got shape {}".format(shape_tensors))
                actual_pred = tensorflow.gather(y_pred, step)
                actual_true = tensorflow.gather(y_true, step)

                pred_final_out = calculates_predicted_words_specific_frame(word_vectors_prediction=actual_pred,
                                                                           target_word2vec=self.target_word_vectors,
                                                                           output_vector_tensors=True)

                # pred_final_out = tensorflow.constant(pred_final_out, dtype=tensorflow.float32)
                tf_equals = tensorflow.reduce_all(tensorflow.equal(pred_final_out, actual_true), axis=-1)
                equals_number = tensorflow.math.count_nonzero(tf_equals)
                v = tensorflow.concat([v, [tensorflow.divide(
                    tensorflow.cast(equals_number, dtype=tensorflow.float32),
                    tensorflow.cast(shape_tensors[-2], dtype=tensorflow.float32)
                )]], 0)
                return v, tensorflow.add(step, 1)

            counter = tensorflow.constant(0, dtype=tensorflow.int32)
            values, _ = tensorflow.while_loop(
                cond=lambda _, count: count < tensorflow.shape(y_true)[0],
                body=process_batch,
                loop_vars=[tensorflow.zeros((0,), dtype=tensorflow.float32),
                           counter],
                shape_invariants=[tensorflow.TensorShape((None,)), counter.get_shape()],
                parallel_iterations=5
            )
            # sys.stderr.write("Calculates {}".format(tensorflow.shape(values)[0]))
            # values = tensorflow.slice(values, [1], tensorflow.subtract(tensorflow.shape(values), 1))
        if sample_weight is not None:
            sample_weight = tensorflow.cast(sample_weight, self.dtype)
            values = tensorflow.multiply(values, sample_weight)

        self.metric_value_sum.assign_add(tensorflow.reduce_sum(values))
        self.metric_value_count.assign_add(tensorflow.shape(values)[0])

    def result(self):
        try:
            return self.metric_value_sum / tensorflow.cast(self.metric_value_count, tensorflow.float32)
        except ZeroDivisionError:
            return 0

    def reset_states(self):
        self.metric_value_sum.assign(0)
        self.metric_value_count.assign(0)
