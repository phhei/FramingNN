from typing import Optional

import torch
from loguru import logger

dict_activation = {
    "Linear": torch.nn.Identity,
    "ReLU": torch.nn.ReLU,
    "GeLU": torch.nn.GELU,
    "Sigmoid": torch.nn.Sigmoid,
    "TanH": torch.nn.Tanh
}


class LinearNNEncoder(torch.nn.Module):
    """
    see https://github.com/phhei/SemEval233FramingPublic
    """
    def __init__(self, activation_module: Optional[str] = None, in_features: int = 300, out_features: int = 300) -> None:
        super().__init__()

        self.in_layer = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        self.activation_module = dict_activation[activation_module]() if activation_module else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _single_sample_forward(tensor_for_single_sample_2d: torch.Tensor) -> torch.Tensor:
            conv_in_tensors = []

            for tensor_slice in tensor_for_single_sample_2d:
                if torch.all(tensor_slice == -1.).item():
                    logger.trace("Detected padding slice -- ignore")
                else:
                    conv_in_tensors.append(
                        self.activation_module(self.in_layer(tensor_slice))
                    )

            logger.debug("Encoded {} tensor slices (e.g. word embeddings)", len(conv_in_tensors))

            if len(conv_in_tensors) == 1:
                logger.warning("Only a single slice (one token in this sample/ text split) - "
                               "we can't calculate a derivation!")
                mean = conv_in_tensors[0]
                std = torch.zeros_like(mean)
            else:
                std, mean = torch.std_mean(torch.stack(conv_in_tensors, dim=0), dim=0)
            logger.trace("Combining std({}) and mean({}) now", std.cpu().tolist(), mean.cpu().tolist())
            std_mean = torch.concat((std, mean), dim=0)

            return std_mean

        predictions = []

        for _x in x:
            predictions.append(
                _single_sample_forward(tensor_for_single_sample_2d=_x)
            )

        logger.trace("Combine {} predictions now", len(predictions))

        stacked_predictions = torch.stack(predictions, dim=0)
        if torch.any(torch.isnan(stacked_predictions)).item():
            logger.warning("This prediction-tensor is partially invalid: {}", stacked_predictions)
            stacked_predictions = \
                torch.masked_fill(stacked_predictions, mask=torch.isnan(stacked_predictions), value=0.)

        return stacked_predictions
