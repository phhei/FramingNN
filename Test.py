"""
OBSOLETE
"""

import numpy
import tensorflow
from keras import Model
from keras import layers
from keras import losses


# of course, I can define here two different loss functions - distributing them among the tasks
def soft_parameter_sharing(layer1: layers.Dense, layer2: layers.Dense):
    def loss(y_true, y_pred):
        print("Shape TRUE: {}, shape PRED {}".format(y_true.shape, y_pred.shape))
        # y_true and y_pred are <class 'tensorflow.python.framework.ops.Tensor'> - this function will get a batch. For every sample in batch, you have one y-tensor, hence (batch, reimining_deminsion)
        # should return a tensor of shape (batch,)
        # soft parameter sharing: its not so important, that the bias parameters of the two layers are the same (weight: 0.01)
        weights_neurons_1 = tensorflow.reshape(layer1.trainable_weights[0], [-1])
        #tensorflow.print(weights_neurons_1)
        weights_bias_1 = tensorflow.reshape(layer1.trainable_weights[1], [-1])
        weights_neurons_2 = tensorflow.reshape(layer2.trainable_weights[0], [-1])
        weights_bias_2 = tensorflow.reshape(layer2.trainable_weights[1], [-1])
        # alternative = tensorflow.reduce_sum(tensorflow.abs(tensorflow.subtract(weights_neurons_1, weights_neurons_2)), axis=1) + ... <-- scsles the first vector!
        ret = losses.mean_absolute_error(y_true=y_true, y_pred=y_pred) +\
              tensorflow.reduce_sum(tensorflow.abs(tensorflow.subtract(weights_neurons_1, weights_neurons_2))) +\
              tensorflow.scalar_mul(0.01, tensorflow.reduce_sum(tensorflow.abs(tensorflow.subtract(weights_bias_1, weights_bias_2))))
        return ret

    return loss


if __name__ == "__main__":
    # our both inputs - can be different shapes
    input1 = layers.Input(shape=(4,2), name="input1")
    input2 = layers.Input(shape=(4,2), name="input2")
    # train / process on both tasks - This uses the same layer for both sides. (Weighs and bias are shared)
    main = layers.GRU(units=2, name="main")
    middle_input1 = main(input1)
    middle_input2 = main(input2)
    # again, we fork the flow
    out1 = layers.Dense(units=1, activation="sigmoid", name="task1")(middle_input1)
    out2 = layers.Dense(units=10, activation="relu", name="task2")(middle_input2)
    model = Model(inputs=(input1, input2), outputs=(out1, out2))

    model.compile(optimizer="adam",
                  loss={"task1": soft_parameter_sharing(main, main), "task2": soft_parameter_sharing(main, main)},
                  metrics={"task1": "binary_accuracy", "task2": "LogCoshError"})

    model.summary()

    #print(main1.get_weights())
    #print(main2.get_weights())

    # we must take care that the sample size of both tasks must be equal - we can oversample for the smaller dataset ;)
    model.fit(x={"input1": numpy.zeros(shape=(10, 4, 2), dtype="float32"),
                 "input2": numpy.ones(shape=(10, 4, 2), dtype="float32")},
              y={"task1": numpy.ones(shape=(10, 1), dtype="float32"),
                 "task2": numpy.zeros(shape=(10, 10), dtype="float32")},
              epochs=30, batch_size=2, verbose=2)

    #print(main1.get_weights())
    #print(main2.get_weights())
