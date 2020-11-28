from typing import List

import loguru
import tensorflow
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils

logger = loguru.logger


# noinspection PyBroadException
class SoftParameterSharingLoss(Loss):
    def __init__(self, base_loss: Loss, shared_layers: List[Layer], impact_value: float, scalar_value=1.0, name=None):
        """
        A Loss function which enables the soft parameter sharing among two tasks regarding a selection of layers
        :param base_loss: the base loss (calculates the relevant loss between ``y_pred`` and ``y_true``
        :param shared_layers: a list of layers which should share parameters
        :param impact_value: the lambda (how crucial s the parameter sharing)
        :param scalar_value: a value to scale the final summed error
        :param name:
        """
        super().__init__(
            name="{}_shared".format(base_loss.name) if name is None else name, reduction=losses_utils.ReductionV2.AUTO
        )
        self.base_loss = base_loss
        self.shared_layers = shared_layers
        self.impact_value = impact_value
        self.scalar_value = scalar_value

    def call(self, y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor):
        base_err = self.base_loss(y_true=y_true, y_pred=y_pred)
        shared_err = tensorflow.zeros(shape=(1,), dtype=y_true.dtype)
        if self.impact_value != 0.0:
            try:
                for weights_left, weights_right in zip(*[s_l.trainable_weights for s_l in self.shared_layers]):
                    # squared https://de.wikipedia.org/wiki/Frobeniusnorm
                    try:
                        shared_err = tensorflow.add(shared_err,
                                                    tensorflow.reduce_sum(
                                                        tensorflow.reduce_sum(
                                                            tensorflow.pow(
                                                                tensorflow.subtract(weights_left, weights_right),
                                                                2
                                                            )
                                                            , axis=-1
                                                        )
                                                        , axis=-1
                                                    ))
                        logger.trace("Shared_err: {}", shared_err)
                    except ValueError:
                        logger.debug("Empty weight matrix - no shared error to calculate (occurs sometimes with LSTMs)")
            except ValueError:
                logger.exception("Until yet, only two layers are allowed to share but you want to share weights among"
                                 " {}", len(self.shared_layers))
            except Exception:
                logger.exception("Something went wrong while calculating the shared weight value of {}",
                                 self.shared_layers)

        try:
            if shared_err.dtype != base_err.dtype:
                shared_err = tensorflow.cast(shared_err, dtype=base_err.dtype)
            if shared_err.dtype != tensorflow.float32:
                w = tensorflow.constant(self.impact_value, dtype=shared_err.dtype)
                s = tensorflow.constant(self.scalar_value, dtype=shared_err.dtype)
            else:
                w = self.impact_value
                s = self.scalar_value
            return tensorflow.multiply(tensorflow.add(base_err, tensorflow.multiply(shared_err, w)), s)
        except Exception:
            logger.exception("Convert error. Maybe you have strange initialization values."
                             "Hence, we return only the plain error loss")
            return base_err


class ScalingLoss(Loss):
    def __init__(self, base_loss: Loss, scale_factor: float, reduction=losses_utils.ReductionV2.AUTO, name=None):
        """
        A class to avoid a training weight bias in multi-task-learning due to up-sampled datasets which have an unequal
        length. Because: without a scale, the samples of the smaller dataset will be iterate multiple times in one epoch
        If each iteration counts full, it's more likely that the learner will try to optimize for the task of the
        smaller dataset. Additionally, for the smaller dataset, there are no "single" epochs anymore, but a batch of
        epochs. The fine-granularity is lost!
        :param base_loss: the basic loss which you want to scale
        :param scale_factor: the factor. Should be assigned to smaller dataset: smaller_dataset/ bigger_dataset
        :param reduction: see Loss-Base-class
        :param name: name
        """
        super().__init__(reduction=reduction, name="{}_scaled".format(base_loss.name) if name is None else name)
        self.base_loss = base_loss
        self.scale_factor = scale_factor

    def call(self, y_true, y_pred):
        base_err = self.base_loss(y_true=y_true, y_pred=y_pred)
        if base_err.dtype != tensorflow.float32:
            s = tensorflow.constant(self.scale_factor, dtype=base_err.dtype)
        else:
            s = self.scale_factor

        return tensorflow.multiply(base_err, s)
